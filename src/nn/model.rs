//! 価値ネットワーク：局面（手番側視点）→ 手番側の勝率
//!
//! 構成は 2 種類（[`Architecture`]）。出力はロジットで、勝率はシグモイドを通して得る。
//! - `Small`：3×3 畳み込み × 3 層（32 チャンネル, ReLU）→ 全結合 64 → 全結合 1
//! - `ResNet`：3×3 畳み込み（stem）→ 残差ブロック × `blocks`（各 3×3 畳み込み 2 層 + バッチ正規化）
//!   → 1×1 畳み込み（4 チャンネル）→ 全結合 128 → 全結合 1
//!
//! 構成はモデルファイルのメタデータに保存し、読み込み時に自動で判別する。
//! メタデータのない古いファイルは `Small` として読み込む。

use std::fmt;
use std::path::Path;

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::store::burn_pack::Reader;
use burn::store::{BurnpackStore, ModuleSnapshot};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{Device, Tensor, TensorData};

use super::features::{FEATURE_LEN, PLANES};

/// ネットワークの構成
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// 小さな CNN（最初に学習したモデルの構成）
    Small,
    /// 残差ネットワーク
    ResNet { channels: usize, blocks: usize },
}

impl Architecture {
    /// 大きなネットワークの既定の構成
    pub const LARGE: Architecture = Architecture::ResNet { channels: 64, blocks: 6 };

    /// `small`、`resnet`（= 既定の大きさ）、`resnet:CHANNELSxBLOCKS`（例 `resnet:128x10`）を解釈する
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "small" => Some(Architecture::Small),
            "resnet" | "large" => Some(Self::LARGE),
            _ => {
                let (c, b) = s.strip_prefix("resnet:")?.split_once('x')?;
                let (channels, blocks) = (c.parse().ok()?, b.parse().ok()?);
                (channels > 0).then_some(Architecture::ResNet { channels, blocks })
            }
        }
    }
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Architecture::Small => write!(f, "small"),
            Architecture::ResNet { channels, blocks } => write!(f, "resnet:{channels}x{blocks}"),
        }
    }
}

/// モデルファイルのメタデータに構成を保存するキー
const ARCHITECTURE_KEY: &str = "othello.architecture";

fn conv3x3(input: usize, output: usize, device: &Device) -> Conv2d {
    Conv2dConfig::new([input, output], [3, 3]).with_padding(PaddingConfig2d::Same).init(device)
}

/// 小さな CNN（`Architecture::Small`）
#[derive(Module, Debug)]
pub struct SmallNet {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    fc1: Linear,
    fc2: Linear,
}

const SMALL_CHANNELS: usize = 32;
const SMALL_HIDDEN: usize = 64;

impl SmallNet {
    fn new(device: &Device) -> Self {
        SmallNet {
            conv1: conv3x3(PLANES, SMALL_CHANNELS, device),
            conv2: conv3x3(SMALL_CHANNELS, SMALL_CHANNELS, device),
            conv3: conv3x3(SMALL_CHANNELS, SMALL_CHANNELS, device),
            fc1: LinearConfig::new(SMALL_CHANNELS * 64, SMALL_HIDDEN).init(device),
            fc2: LinearConfig::new(SMALL_HIDDEN, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let x = relu(self.conv1.forward(x));
        let x = relu(self.conv2.forward(x));
        let x = relu(self.conv3.forward(x));
        let n = x.dims()[0];
        let x = x.reshape([n, SMALL_CHANNELS * 64]);
        self.fc2.forward(relu(self.fc1.forward(x)))
    }
}

/// 残差ブロック：x + BN(conv(ReLU(BN(conv(x))))) → ReLU
#[derive(Module, Debug)]
pub struct ResBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
}

impl ResBlock {
    fn new(channels: usize, device: &Device) -> Self {
        ResBlock {
            conv1: conv3x3(channels, channels, device),
            bn1: BatchNormConfig::new(channels).init(device),
            conv2: conv3x3(channels, channels, device),
            bn2: BatchNormConfig::new(channels).init(device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let y = relu(self.bn1.forward(self.conv1.forward(x.clone())));
        let y = self.bn2.forward(self.conv2.forward(y));
        relu(x + y)
    }
}

/// 価値ヘッドの 1×1 畳み込みの出力チャンネル数
const HEAD_CHANNELS: usize = 4;
/// 価値ヘッドの全結合の隠れ層の幅
const HEAD_HIDDEN: usize = 128;

/// 残差ネットワーク（`Architecture::ResNet`）
#[derive(Module, Debug)]
pub struct ResNet {
    stem: Conv2d,
    stem_bn: BatchNorm,
    blocks: Vec<ResBlock>,
    head_conv: Conv2d,
    head_bn: BatchNorm,
    fc1: Linear,
    fc2: Linear,
}

impl ResNet {
    fn new(channels: usize, blocks: usize, device: &Device) -> Self {
        ResNet {
            stem: conv3x3(PLANES, channels, device),
            stem_bn: BatchNormConfig::new(channels).init(device),
            blocks: (0..blocks).map(|_| ResBlock::new(channels, device)).collect(),
            head_conv: Conv2dConfig::new([channels, HEAD_CHANNELS], [1, 1]).init(device),
            head_bn: BatchNormConfig::new(HEAD_CHANNELS).init(device),
            fc1: LinearConfig::new(HEAD_CHANNELS * 64, HEAD_HIDDEN).init(device),
            fc2: LinearConfig::new(HEAD_HIDDEN, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let mut x = relu(self.stem_bn.forward(self.stem.forward(x)));
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = relu(self.head_bn.forward(self.head_conv.forward(x)));
        let n = x.dims()[0];
        let x = x.reshape([n, HEAD_CHANNELS * 64]);
        self.fc2.forward(relu(self.fc1.forward(x)))
    }

    fn architecture(&self) -> Architecture {
        // 畳み込みの重みの形は [出力チャンネル, 入力チャンネル, 縦, 横]
        let [channels, _, _, _] = self.stem.weight.shape().dims();
        Architecture::ResNet { channels, blocks: self.blocks.len() }
    }
}

// モデルは数個しか作らず値渡しも少ないので、バリアント間のサイズ差は問題にならない
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum ValueNet {
    Small(SmallNet),
    ResNet(ResNet),
}

impl ValueNet {
    /// 重みをランダムに初期化したネットワークを作る
    pub fn new(architecture: Architecture, device: &Device) -> Self {
        match architecture {
            Architecture::Small => ValueNet::Small(SmallNet::new(device)),
            Architecture::ResNet { channels, blocks } => ValueNet::ResNet(ResNet::new(channels, blocks, device)),
        }
    }

    pub fn architecture(&self) -> Architecture {
        match self {
            ValueNet::Small(_) => Architecture::Small,
            ValueNet::ResNet(net) => net.architecture(),
        }
    }

    /// 入力 `[N, PLANES, 8, 8]` → ロジット `[N, 1]`
    pub fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        match self {
            ValueNet::Small(net) => net.forward(x),
            ValueNet::ResNet(net) => net.forward(x),
        }
    }

    /// 特徴量（`n` 局面分を連結したもの）から勝率を一括で求める
    pub fn predict(&self, features: Vec<f32>, n: usize, device: &Device) -> Vec<f32> {
        assert_eq!(features.len(), n * FEATURE_LEN);
        let x = Tensor::<4>::from_data(TensorData::new(features, [n, PLANES, 8, 8]), device);
        sigmoid(self.forward(x)).into_data().try_to_vec::<f32>().expect("f32 output")
    }

    /// パラメータ数
    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }

    /// 保存する。テンソル名は中身の構造体のもの（`Small` は以前の形式と同じ）で、構成はメタデータに記録する。
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let mut store = BurnpackStore::from_file(path)
            .overwrite(true)
            .auto_extension(false)
            .metadata(ARCHITECTURE_KEY, self.architecture().to_string());
        // 学習中のモデルでは、バッチ正規化の移動平均の最新の更新が `valid()` を通すまで反映されないため、
        // 推論用の形に変換してから保存する（推論用のモデルならそのまま）
        let model = self.valid();
        let result = match &model {
            ValueNet::Small(net) => net.save_into(&mut store),
            ValueNet::ResNet(net) => net.save_into(&mut store),
        };
        result.map_err(|e| format!("{}: {e}", path.display()))
    }

    /// モデルファイルに記録された構成を読む（記録のない古いファイルは `Small`）
    pub fn read_architecture(path: impl AsRef<Path>) -> Result<Architecture, String> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(format!("{} が見つかりません", path.display()));
        }
        let reader = Reader::from_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
        match reader.metadata().get(ARCHITECTURE_KEY) {
            None => Ok(Architecture::Small),
            Some(s) => Architecture::parse(s).ok_or_else(|| format!("{}: 不明な構成 {s}", path.display())),
        }
    }

    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self, String> {
        let path = path.as_ref();
        let mut model = ValueNet::new(Self::read_architecture(path)?, device);
        let mut store = BurnpackStore::from_file(path).auto_extension(false);
        let result = match &mut model {
            ValueNet::Small(net) => net.load_from(&mut store),
            ValueNet::ResNet(net) => net.load_from(&mut store),
        };
        let applied = result.map_err(|e| format!("{}: {e}", path.display()))?;
        if !applied.missing.is_empty() {
            return Err(format!("{}: 重みが足りません（{} 個）", path.display(), applied.missing.len()));
        }
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::nn::features::features;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("othello-nn-test-{}-{name}", std::process::id()))
    }

    #[test]
    fn architecture_parse_and_display_roundtrip() {
        for arch in [Architecture::Small, Architecture::LARGE, Architecture::ResNet { channels: 128, blocks: 10 }] {
            assert_eq!(Architecture::parse(&arch.to_string()), Some(arch));
        }
        assert_eq!(Architecture::parse("resnet"), Some(Architecture::LARGE));
        assert_eq!(Architecture::parse("resnet:0x3"), None);
        assert_eq!(Architecture::parse("huge"), None);
    }

    #[test]
    fn save_and_load_roundtrip_for_each_architecture() {
        let device = Device::flex();
        let small_resnet = Architecture::ResNet { channels: 8, blocks: 2 };
        for arch in [Architecture::Small, small_resnet] {
            let model = ValueNet::new(arch, &device);
            let path = temp_path(&format!("{arch}.bpk").replace(':', "_"));
            model.save(&path).unwrap();
            assert_eq!(ValueNet::read_architecture(&path).unwrap(), arch);
            let loaded = ValueNet::load(&path, &device).unwrap();
            assert_eq!(loaded.architecture(), arch);

            let f = features(&Board::new());
            let a = model.predict(f.clone(), 1, &device);
            let b = loaded.predict(f, 1, &device);
            assert!((a[0] - b[0]).abs() < 1e-6, "{arch}: {a:?} vs {b:?}");
            let _ = std::fs::remove_file(path);
        }
    }

    /// バッチ正規化の移動平均（学習中に更新される）も保存・復元される
    #[test]
    fn batch_norm_running_stats_are_saved() {
        let device = Device::flex().autodiff();
        let model = ValueNet::new(Architecture::ResNet { channels: 8, blocks: 1 }, &device);
        // 学習モードで何度か順伝播すると移動平均が更新される
        let x = Tensor::<4>::from_data(TensorData::new(features(&Board::new()).repeat(4), [4, PLANES, 8, 8]), &device);
        for _ in 0..5 {
            let _ = model.forward(x.clone());
        }
        let path = temp_path("bn.bpk");
        model.save(&path).unwrap();

        let cpu = Device::flex();
        let loaded = ValueNet::load(&path, &cpu).unwrap();
        let expected = model.valid().predict(features(&Board::new()), 1, &cpu);
        let actual = loaded.predict(features(&Board::new()), 1, &cpu);
        assert!((expected[0] - actual[0]).abs() < 1e-5, "{expected:?} vs {actual:?}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn existing_small_model_file_still_loads() {
        // 既存の学習済みモデル（メタデータなし）が読める
        let path = Path::new("runs/default/latest.bpk");
        if !path.exists() {
            return;
        }
        assert_eq!(ValueNet::read_architecture(path).unwrap(), Architecture::Small);
        let model = ValueNet::load(path, &Device::flex()).unwrap();
        assert_eq!(model.architecture(), Architecture::Small);
    }

    #[test]
    fn large_network_has_more_parameters() {
        let device = Device::flex();
        let small = ValueNet::new(Architecture::Small, &device).num_params();
        let large = ValueNet::new(Architecture::LARGE, &device).num_params();
        assert!(large > small * 2, "small {small}, large {large}");
    }

    #[test]
    fn load_reports_missing_file() {
        assert!(ValueNet::load("/nonexistent/model.bpk", &Device::flex()).is_err());
    }
}
