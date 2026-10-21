//! 対局者の設定。アルゴリズムごとに持つパラメータが異なる。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::nn::device::GPU_BATCH_PER_THREAD;
use crate::nn::{inference_device, load_evaluator};
use crate::ai::{
    Ai, Evaluator, Mcts, MctsConfig, ParallelMode, PlayoutEvaluator, PositionalEvaluator, RandomAi, SearchLimit,
};

#[derive(Debug, Clone, PartialEq)]
pub struct HumanParams {
    /// 自分の手番中に、相手 AI が先読み（ponder）することを許可するか
    pub allow_ponder: bool,
}

impl Default for HumanParams {
    fn default() -> Self {
        HumanParams { allow_ponder: true }
    }
}

/// MCTS の葉ノードの評価関数の種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluatorChoice {
    /// ランダムプレイアウト（回数は `MctsParams::playouts_per_eval`）
    Playout,
    /// 静的評価関数（マスの重み + 着手可能数）
    Positional,
    /// 価値ネットワーク（`MctsParams::model_path` のモデルを、GPU が使えれば GPU、なければ CPU で推論）
    Nn,
}

/// NN モデルの既定のパス（同梱の学習済みモデル。resnet 64x6）。
/// 相対パスは [`resolve_model_path`] で、作業ディレクトリになければ実行ファイルの場所から探す。
pub const DEFAULT_MODEL_PATH: &str = "models/large.bpk";

/// モデルファイルのパスを解決する。相対パスで作業ディレクトリに見つからなければ、
/// 実行ファイルと同じディレクトリからの相対パスとして探す（配布版をどこから起動しても同梱のモデルを使えるように）。
/// どちらにもなければ指定どおりのパスを返す。
pub fn resolve_model_path(path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() || path.exists() {
        return path;
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(&path)))
        .filter(|p| p.exists())
        .unwrap_or(path)
}

#[derive(Debug, Clone, PartialEq)]
pub struct MctsParams {
    /// 1 手あたりの思考時間（秒）
    pub think_time_secs: f64,
    /// 葉ノードの評価関数
    pub evaluator: EvaluatorChoice,
    /// プレイアウト評価時のプレイアウト回数
    pub playouts_per_eval: u32,
    /// NN 評価時のモデルファイル
    pub model_path: String,
    /// NN 評価時に、1 スレッドが 1 回の推論にまとめる葉の数（木の並列化のときだけ有効）。
    /// `None` なら推論デバイスに合わせて自動で決める（[`auto_nn_batch_size`]）。
    pub nn_batch_size: Option<u32>,
    /// 探索スレッド数
    pub threads: u32,
    /// 並列化方式
    pub parallel: ParallelMode,
    /// 探索木を次の手に再利用するか（ponder にも必要）
    pub reuse_tree: bool,
}

/// NN 評価時の、1 スレッドが 1 回の推論にまとめる葉の数の自動設定値。
///
/// GPU は [`GPU_BATCH_PER_THREAD`]（推論デバイスの確認もこの大きさで行う）。
/// CPU は小さなモデルなら K=4 前後で 2 倍ほど速くなるが、大きなモデル（resnet）では逆に遅くなるため 1 にする。
pub fn auto_nn_batch_size(is_gpu: bool) -> u32 {
    if is_gpu { GPU_BATCH_PER_THREAD } else { 1 }
}

/// この PC の論理 CPU 数（スレッド数の上限・初期値に使う）
pub fn available_threads() -> u32 {
    std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1)
}

impl Default for MctsParams {
    fn default() -> Self {
        MctsParams {
            think_time_secs: 5.0,
            evaluator: EvaluatorChoice::Playout,
            playouts_per_eval: 1,
            model_path: DEFAULT_MODEL_PATH.to_string(),
            nn_batch_size: None,
            threads: available_threads(),
            parallel: ParallelMode::Tree,
            reuse_tree: true,
        }
    }
}

impl MctsParams {
    pub fn to_config(&self) -> MctsConfig {
        MctsConfig {
            threads: self.threads.max(1),
            parallel: self.parallel,
            reuse_tree: self.reuse_tree,
            // まとめて評価して速くなるのは NN だけなので、他の評価関数では 1 葉ずつ評価する
            batch_size: if self.evaluator == EvaluatorChoice::Nn { self.effective_nn_batch_size() } else { 1 },
            limit: SearchLimit::Time(Duration::from_secs_f64(self.think_time_secs)),
            ..MctsConfig::default()
        }
    }

    /// NN 評価時に実際に使う、1 回の推論にまとめる葉の数（自動なら推論デバイスを確認して決める）
    pub fn effective_nn_batch_size(&self) -> u32 {
        match self.nn_batch_size {
            Some(n) => n.max(1),
            None => auto_nn_batch_size(inference_device().is_gpu),
        }
    }

    /// 評価関数を作る。NN のモデルが読み込めなければエラー。
    pub fn build_evaluator(&self) -> Result<Arc<dyn Evaluator>, String> {
        Ok(match self.evaluator {
            EvaluatorChoice::Playout => Arc::new(PlayoutEvaluator::new(self.playouts_per_eval)),
            EvaluatorChoice::Positional => Arc::new(PositionalEvaluator::default()),
            EvaluatorChoice::Nn => {
                // GPU で推論する場合、全スレッドの葉が 1 回の推論にまとまるようにする
                let max_batch = self.threads.max(1) as usize * self.effective_nn_batch_size() as usize;
                load_evaluator(resolve_model_path(&self.model_path), max_batch)?
            }
        })
    }
}

/// 対局者の種類とそのパラメータ
#[derive(Debug, Clone, PartialEq)]
pub enum PlayerKind {
    Human(HumanParams),
    Random,
    Mcts(MctsParams),
}

impl PlayerKind {
    /// 選択肢の一覧（各種類のデフォルト設定）
    pub fn choices() -> Vec<PlayerKind> {
        vec![PlayerKind::Human(HumanParams::default()), PlayerKind::Random, PlayerKind::Mcts(MctsParams::default())]
    }

    pub fn label(&self) -> &'static str {
        match self {
            PlayerKind::Human(_) => "人間",
            PlayerKind::Random => "ランダム",
            PlayerKind::Mcts(_) => "MCTS",
        }
    }

    pub fn is_human(&self) -> bool {
        matches!(self, PlayerKind::Human(_))
    }

    /// 人間の手番中に、相手 AI の ponder を許可しているか（AI 同士では常に `false`）
    pub fn allows_opponent_ponder(&self) -> bool {
        matches!(self, PlayerKind::Human(p) if p.allow_ponder)
    }

    /// ponder できる AI か
    pub fn can_ponder(&self) -> bool {
        matches!(self, PlayerKind::Mcts(p) if p.reuse_tree)
    }

    /// 思考時間の上限（進捗表示用）。時間制限のないものは `None`。
    pub fn think_time(&self) -> Option<Duration> {
        match self {
            PlayerKind::Mcts(p) => Some(Duration::from_secs_f64(p.think_time_secs)),
            _ => None,
        }
    }

    /// AI を生成する。人間の場合は `Ok(None)`。評価関数の準備に失敗したらエラー。
    pub fn build_ai(&self) -> Result<Option<Box<dyn Ai + Send>>, String> {
        Ok(match self {
            PlayerKind::Human(_) => None,
            PlayerKind::Random => Some(Box::new(RandomAi::new())),
            PlayerKind::Mcts(p) => Some(Box::new(Mcts::with_evaluator(p.to_config(), p.build_evaluator()?))),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::ValueNet;
    use crate::nn::features::{FEATURE_LEN, write_features};

    /// 同梱モデルが既定のパスにあり、読み込んで推論できる（モデルを差し替えた場合の確認も兼ねる。構成は問わない）
    #[test]
    fn bundled_model_is_the_default_and_loads() {
        let path = resolve_model_path(DEFAULT_MODEL_PATH);
        assert!(path.exists(), "{}", path.display());
        let device = burn::tensor::Device::flex();
        let model = ValueNet::load(&path, &device).unwrap();
        let mut features = vec![0.0; FEATURE_LEN];
        write_features(&crate::board::Board::new(), &mut features);
        let value = model.predict(features, 1, &device)[0];
        assert!((0.0..=1.0).contains(&value), "{value}");
    }

    #[test]
    fn missing_relative_path_is_returned_as_is() {
        assert_eq!(resolve_model_path("no/such/model.bpk"), PathBuf::from("no/such/model.bpk"));
    }
}
