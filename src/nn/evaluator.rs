//! 価値ネットワークによる MCTS の評価関数
//!
//! - [`NnEvaluator`]：呼び出したスレッドで推論する。CPU 向け。
//! - [`BatchedNnEvaluator`]：複数スレッドからの評価要求を 1 つのバッチにまとめて推論する。
//!   GPU は 1 局面ずつだと遅いが、まとめて流すと非常に速いため、多数の対局を並行させる自己対戦や、
//!   葉をまとめて評価する MCTS（`batch_size`）で使う。
//!
//! どちらも [`Evaluator::evaluate_batch`] で複数局面を 1 回の推論にまとめられる。
//! [`load_evaluator`] は、使えるデバイス（GPU があれば GPU）に合わせてどちらかを作る。

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use burn::tensor::Device;

use super::device::{inference_device, inference_device_if_ready};
use super::features::{FEATURE_LEN, write_features};
use super::model::ValueNet;
use crate::ai::Evaluator;
use crate::board::Board;
use crate::rng::Rng;

/// 表示用のデバイス名
fn device_label(device: &Device) -> String {
    match inference_device_if_ready() {
        Some(d) if d.device == *device => d.label.clone(),
        _ if *device == Device::flex() => "CPU".to_string(),
        _ => "GPU".to_string(),
    }
}

/// `boards` の特徴量を連結する
fn batch_features(boards: &[Board]) -> Vec<f32> {
    let mut features = vec![0.0; boards.len() * FEATURE_LEN];
    for (board, f) in boards.iter().zip(features.as_chunks_mut::<FEATURE_LEN>().0) {
        write_features(board, f);
    }
    features
}

/// モデルを読み込み、推論デバイス（[`inference_device`]：GPU が使えれば GPU、なければ CPU）で
/// 推論する評価関数を作る。
///
/// GPU では評価要求をまとめる [`BatchedNnEvaluator`]（1 回の推論は最大 `max_batch` 局面）、
/// CPU では呼び出したスレッドで推論する [`NnEvaluator`] になる。
pub fn load_evaluator(path: impl AsRef<std::path::Path>, max_batch: usize) -> Result<Arc<dyn Evaluator>, String> {
    let path = path.as_ref();
    let target = inference_device();
    let model = ValueNet::load(path, &target.device)?;
    let name = format!("nn {}({})", model.architecture(), path.display());
    let device = target.device.clone();
    Ok(if target.is_gpu {
        Arc::new(BatchedNnEvaluator::new(model, device, max_batch, GPU_MAX_WAIT, name))
    } else {
        Arc::new(NnEvaluator::new(model, device, name))
    })
}

/// [`load_evaluator`] の GPU 推論で、バッチが埋まるのを待つ最大時間
const GPU_MAX_WAIT: Duration = Duration::from_millis(2);

/// 呼び出したスレッドで推論する評価関数
pub struct NnEvaluator {
    model: ValueNet,
    device: Device,
    device_label: String,
    name: String,
}

impl NnEvaluator {
    pub fn new(model: ValueNet, device: Device, name: impl Into<String>) -> Self {
        let device_label = device_label(&device);
        NnEvaluator { model, device, device_label, name: name.into() }
    }
}

impl Evaluator for NnEvaluator {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn evaluate(&self, board: &Board, _rng: &mut Rng) -> f64 {
        let mut f = vec![0.0; FEATURE_LEN];
        write_features(board, &mut f);
        self.model.predict(f, 1, &self.device)[0] as f64
    }

    fn evaluate_batch(&self, boards: &[Board], _rng: &mut Rng) -> Vec<f64> {
        if boards.is_empty() {
            return Vec::new();
        }
        let values = self.model.predict(batch_features(boards), boards.len(), &self.device);
        values.into_iter().map(f64::from).collect()
    }

    fn device(&self) -> Option<&str> {
        Some(&self.device_label)
    }
}

/// 推論スレッドの数
pub const INFERENCE_WORKERS: usize = 2;

/// バッチを締め切る、要求の途切れの長さ
pub const BATCH_GAP: Duration = Duration::from_micros(100);

struct Request {
    features: Vec<f32>,
    reply: SyncSender<f32>,
}

/// バッチ推論の統計
#[derive(Debug, Default)]
pub struct BatchStats {
    pub batches: AtomicU64,
    pub positions: AtomicU64,
    /// 推論（GPU への転送・計算・結果の取得）にかかった時間の合計（ナノ秒）
    pub inference_nanos: AtomicU64,
}

/// 評価要求をバッチにまとめて推論する評価関数。
///
/// 推論は専用スレッド（[`INFERENCE_WORKERS`] 本）で行う。1 本がバッチを推論している間に
/// 別の 1 本が次のバッチの要求を集めることで、GPU の計算と要求の収集・返信を重ねる。
pub struct BatchedNnEvaluator {
    tx: Option<Sender<Request>>,
    workers: Vec<JoinHandle<()>>,
    stats: Arc<BatchStats>,
    device_label: String,
    name: String,
}

impl BatchedNnEvaluator {
    /// `max_batch`：1 回に推論する最大局面数。
    /// `max_wait`：最初の要求が来てから、バッチが埋まるのを待つ最大時間。
    /// 要求が [`BATCH_GAP`] の間途切れたら、`max_wait` 前でもその時点の要求で推論する。
    pub fn new(model: ValueNet, device: Device, max_batch: usize, max_wait: Duration, name: impl Into<String>) -> Self {
        let device_label = device_label(&device);
        let (tx, rx) = mpsc::channel::<Request>();
        let rx = Arc::new(Mutex::new(rx));
        let stats = Arc::new(BatchStats::default());
        let workers = (0..INFERENCE_WORKERS)
            .map(|_| {
                let (model, device, rx, stats) = (model.clone(), device.clone(), rx.clone(), stats.clone());
                thread::spawn(move || serve(&model, &device, &rx, max_batch.max(1), max_wait, &stats))
            })
            .collect();
        BatchedNnEvaluator { tx: Some(tx), workers, stats, device_label, name: name.into() }
    }

    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }
}

/// 要求を 1 バッチ分集める。送信側が閉じられていれば `false`。
/// 受信口のロックはバッチを集める間だけ持つ（推論中は他の推論スレッドが次のバッチを集められる）。
fn gather(rx: &Mutex<Receiver<Request>>, batch: &mut Vec<Request>, max_batch: usize, max_wait: Duration) -> bool {
    let rx = rx.lock().unwrap();
    let Ok(first) = rx.recv() else { return false };
    batch.push(first);
    let deadline = Instant::now() + max_wait;
    while batch.len() < max_batch {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        match rx.recv_timeout(BATCH_GAP.min(deadline - now)) {
            Ok(req) => batch.push(req),
            Err(RecvTimeoutError::Timeout) => break,
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    true
}

fn serve(
    model: &ValueNet,
    device: &Device,
    rx: &Mutex<Receiver<Request>>,
    max_batch: usize,
    max_wait: Duration,
    stats: &BatchStats,
) {
    let mut batch: Vec<Request> = Vec::with_capacity(max_batch);
    while gather(rx, &mut batch, max_batch, max_wait) {
        // GPU ではテンソルの形が変わるたびにカーネルのコンパイル・チューニングが走って遅くなるため、
        // 常に max_batch 局面分に 0 で埋めて同じ形で推論する
        let n = batch.len();
        let mut features = Vec::with_capacity(max_batch * FEATURE_LEN);
        for req in &batch {
            features.extend_from_slice(&req.features);
        }
        features.resize(max_batch * FEATURE_LEN, 0.0);
        let started = Instant::now();
        let values = model.predict(features, max_batch, device);
        stats.inference_nanos.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        for (req, value) in batch.drain(..).zip(values) {
            // 要求側が既にいなくなっていても問題ない
            let _ = req.reply.send(value);
        }
        stats.batches.fetch_add(1, Ordering::Relaxed);
        stats.positions.fetch_add(n as u64, Ordering::Relaxed);
    }
}

impl Evaluator for BatchedNnEvaluator {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn evaluate(&self, board: &Board, _rng: &mut Rng) -> f64 {
        let mut features = vec![0.0; FEATURE_LEN];
        write_features(board, &mut features);
        let (reply, result) = mpsc::sync_channel(1);
        let tx = self.tx.as_ref().expect("evaluator is alive");
        tx.send(Request { features, reply }).expect("inference thread stopped");
        result.recv().expect("inference thread stopped") as f64
    }

    /// 全局面の要求を続けて送ってから結果を待つ（他のスレッドの要求とも同じバッチにまとまる）
    fn evaluate_batch(&self, boards: &[Board], _rng: &mut Rng) -> Vec<f64> {
        let tx = self.tx.as_ref().expect("evaluator is alive");
        let replies: Vec<Receiver<f32>> = boards
            .iter()
            .map(|board| {
                let mut features = vec![0.0; FEATURE_LEN];
                write_features(board, &mut features);
                let (reply, result) = mpsc::sync_channel(1);
                tx.send(Request { features, reply }).expect("inference thread stopped");
                result
            })
            .collect();
        replies.into_iter().map(|r| r.recv().expect("inference thread stopped") as f64).collect()
    }

    fn device(&self) -> Option<&str> {
        Some(&self.device_label)
    }
}

impl Drop for BatchedNnEvaluator {
    fn drop(&mut self) {
        // 送信側を閉じると推論スレッドが終了する
        self.tx = None;
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Move;

    fn sample_boards() -> Vec<Board> {
        let mut boards = vec![Board::new()];
        for mv in [19, 18, 17, 9, 26, 20] {
            let b = *boards.last().unwrap();
            let next = b.legal_moves().into_iter().find(|m| *m == Move::Place(mv)).unwrap_or(b.legal_moves()[0]);
            boards.push(b.play(next).unwrap());
        }
        boards
    }

    /// まとめて推論しても、1 局面ずつ推論した値と同じ順で同じ値になる
    #[test]
    fn evaluate_batch_matches_evaluate() {
        let device = Device::flex();
        let model = ValueNet::new(crate::nn::model::Architecture::Small, &device);
        let direct = NnEvaluator::new(model.clone(), device.clone(), "direct");
        let batched = BatchedNnEvaluator::new(model, device, 4, Duration::from_millis(5), "batched");
        let boards = sample_boards();
        let mut rng = Rng::new(0);
        let singles: Vec<f64> = boards.iter().map(|b| direct.evaluate(b, &mut rng)).collect();
        for (name, values) in
            [("direct", direct.evaluate_batch(&boards, &mut rng)), ("batched", batched.evaluate_batch(&boards, &mut rng))]
        {
            assert_eq!(values.len(), boards.len(), "{name}");
            for (v, s) in values.iter().zip(&singles) {
                assert!((v - s).abs() < 1e-5, "{name}: {v} vs {s}");
            }
        }
        assert!(direct.evaluate_batch(&[], &mut rng).is_empty());
        assert_eq!(direct.device(), Some("CPU"));
    }

    #[test]
    fn batched_and_direct_evaluators_agree() {
        let device = Device::flex();
        let model = ValueNet::new(crate::nn::model::Architecture::Small, &device);
        let direct = NnEvaluator::new(model.clone(), device.clone(), "direct");
        let batched = Arc::new(BatchedNnEvaluator::new(model, device, 64, Duration::from_millis(5), "batched"));

        let mut boards = vec![Board::new()];
        for mv in [19, 18, 17, 9] {
            let b = *boards.last().unwrap();
            let next = b.legal_moves().into_iter().find(|m| *m == Move::Place(mv)).unwrap_or(b.legal_moves()[0]);
            boards.push(b.play(next).unwrap());
        }

        // 複数スレッドから同時に要求してもそれぞれ正しい値が返る
        let results: Vec<(f64, f64)> = thread::scope(|scope| {
            let handles: Vec<_> = boards
                .iter()
                .map(|b| {
                    let (batched, direct) = (batched.clone(), &direct);
                    scope.spawn(move || {
                        let mut rng = Rng::new(0);
                        (batched.evaluate(b, &mut rng), direct.evaluate(b, &mut rng))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (b, d) in results {
            assert!((b - d).abs() < 1e-5, "batched {b} vs direct {d}");
            assert!((0.0..=1.0).contains(&b));
        }
        assert_eq!(batched.stats().positions.load(Ordering::Relaxed), boards.len() as u64);
    }
}
