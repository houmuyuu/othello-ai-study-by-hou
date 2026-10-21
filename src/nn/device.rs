//! 推論デバイスの選択
//!
//! CUDA 対応（`cuda` フィーチャ）でビルドされていて GPU が使えれば GPU、そうでなければ CPU で推論する。
//! GPU の確認（ドライバの読み込み・テスト計算）は時間がかかるため、初回の 1 回だけ行い結果を使い回す。
//!
//! 確認では、実際の対局と同じ構成のネットワーク・同じ大きさのバッチで推論してみる。
//! - 簡単な計算は通るのに、ネットワークの計算（畳み込みなど）だけが失敗する GPU・ドライバでも、対局の前に CPU に切り替えられる
//! - 初回の推論で行われるカーネルのコンパイルと自動チューニングを済ませておける（対局の最初の手が思考時間を超えない）

use std::sync::OnceLock;

use burn::tensor::Device;

/// 推論に使うデバイスと、その表示用の説明
#[derive(Debug, Clone)]
pub struct InferenceDevice {
    pub device: Device,
    /// GPU か
    pub is_gpu: bool,
    /// 表示名（例：「GPU（CUDA: NVIDIA GeForce RTX 4070）」「CPU」）
    pub label: String,
    /// CPU になった理由（GPU を使う場合は `None`）
    pub fallback_reason: Option<String>,
}

impl InferenceDevice {
    fn cpu(reason: impl Into<String>) -> Self {
        InferenceDevice { device: Device::flex(), is_gpu: false, label: "CPU".to_string(), fallback_reason: Some(reason.into()) }
    }

    /// 理由を含めた 1 行の説明
    pub fn describe(&self) -> String {
        match &self.fallback_reason {
            Some(reason) => format!("{}（{reason}）", self.label),
            None => self.label.clone(),
        }
    }
}

static DEVICE: OnceLock<InferenceDevice> = OnceLock::new();

/// GPU で推論するとき、1 スレッドが 1 回の推論にまとめる葉の数の既定値。
/// GPU は 1 回の推論にかかる固定のコストが大きく、まとめるほど速い（32 スレッドで 1 → 32 にすると反復回数が約 15 倍）。
pub const GPU_BATCH_PER_THREAD: u32 = 32;

/// 推論に使うデバイス。初回の呼び出しで GPU が使えるか確認する（数秒かかることがある）。
pub fn inference_device() -> &'static InferenceDevice {
    DEVICE.get_or_init(|| {
        let d = detect();
        eprintln!("推論デバイス: {}", d.describe());
        d
    })
}

/// 確認が済んでいれば推論デバイスを返す（GUI の表示用。確認中なら `None`）
pub fn inference_device_if_ready() -> Option<&'static InferenceDevice> {
    DEVICE.get()
}

/// GPU が使えるかを確認して推論デバイスを決める
#[cfg(feature = "cuda")]
fn detect() -> InferenceDevice {
    use burn::tensor::{DeviceType, Tensor, TensorData};

    let Some(device) = Device::enumerate(DeviceType::Cuda).first().cloned() else {
        return InferenceDevice::cpu("CUDA の GPU が見つかりません");
    };
    // ドライバや GPU の状態、ライブラリ（NVRTC など）の有無によっては初期化・計算時に panic するため、
    // 別スレッドで試す。内部のスレッドで失敗して応答が返らないこともあるので、待つ時間にも上限を設ける
    let probe = device.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let name = probe.identity().map(|id| id.name);
            let x = Tensor::<1>::from_data(TensorData::new(vec![1.0f32, 2.0], [2]), &probe);
            let y = (x * 2.0).into_data().try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            if y != [2.0, 4.0] {
                return Err(format!("テスト計算の結果が不正です: {y:?}"));
            }
            probe_network(&probe)?;
            Ok(name)
        }));
        let _ = tx.send(result.unwrap_or_else(|panic| Err(panic_message(&*panic))));
    });
    match rx.recv_timeout(PROBE_TIMEOUT) {
        Ok(Ok(name)) => {
            let label = match name {
                Some(name) => format!("GPU（CUDA: {name}）"),
                None => "GPU（CUDA）".to_string(),
            };
            InferenceDevice { device, is_gpu: true, label, fallback_reason: None }
        }
        Ok(Err(e)) => InferenceDevice::cpu(format!("GPU の初期化に失敗: {e}")),
        Err(_) => InferenceDevice::cpu(format!("GPU の初期化が {} 秒以内に終わりません", PROBE_TIMEOUT.as_secs())),
    }
}

/// 既定の設定（スレッド数 = CPU 数、1 スレッドあたり [`GPU_BATCH_PER_THREAD`] 局面）の対局と同じ形で、
/// 既定の構成のネットワークを推論してみる。重みは乱数でよい（カーネルは重みの値によらない）
#[cfg(feature = "cuda")]
fn probe_network(device: &Device) -> Result<(), String> {
    use super::features::FEATURE_LEN;
    use super::model::{Architecture, ValueNet};

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let n = threads * GPU_BATCH_PER_THREAD as usize;
    let started = std::time::Instant::now();
    let model = ValueNet::new(Architecture::LARGE, device);
    let values = model.predict(vec![0.0; n * FEATURE_LEN], n, device);
    if values.len() != n || !values.iter().all(|v| v.is_finite()) {
        return Err(format!("ネットワークの推論結果が不正です（{} 局面中 {} 件）", n, values.len()));
    }
    eprintln!("GPU の準備（{n} 局面の推論）: {:.1} 秒", started.elapsed().as_secs_f64());
    Ok(())
}

/// GPU の確認を待つ最大時間（初回はカーネルのコンパイルと自動チューニングがあるので余裕を持たせる）
#[cfg(feature = "cuda")]
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

#[cfg(feature = "cuda")]
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "不明なエラー".to_string())
}

#[cfg(not(feature = "cuda"))]
fn detect() -> InferenceDevice {
    InferenceDevice::cpu("GPU 非対応のビルド。--features cuda でビルドすると GPU を使えます")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_device_is_usable() {
        let d = inference_device();
        assert_eq!(d.is_gpu, d.fallback_reason.is_none());
        assert!(!d.label.is_empty());
        #[cfg(not(feature = "cuda"))]
        assert!(!d.is_gpu);
        // 選ばれたデバイスで実際に計算できる
        let model = crate::nn::ValueNet::new(crate::nn::Architecture::Small, &d.device);
        let values = model.predict(vec![0.0; 2 * crate::nn::features::FEATURE_LEN], 2, &d.device);
        assert_eq!(values.len(), 2);
    }
}
