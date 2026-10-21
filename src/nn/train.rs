//! 価値ネットワークの学習
//!
//! 損失はロジットに対する二値交差エントロピー（教師値 0, 0.5, 1 をそのまま扱える）。
//! ミニバッチごとに 8 通りの対称変換からランダムに 1 つ選んで適用し、データを水増しする。

use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::activation::softplus;
use burn::tensor::{Device, Tensor, TensorData};

use super::data::Sample;
use super::features::{FEATURE_LEN, PLANES, SYMMETRIES, transform_board, write_features};
use super::model::ValueNet;
use crate::rng::Rng;

#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    /// 検証用に取り分けるデータの割合
    pub validation_fraction: f64,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig { epochs: 4, batch_size: 256, learning_rate: 1e-3, validation_fraction: 0.05, seed: 0 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EpochReport {
    pub epoch: usize,
    pub train_loss: f64,
    pub validation_loss: f64,
    /// 検証データで、予測と結果の勝敗の向き（0.5 より上か下か）が一致した割合（引き分けは除く）
    pub validation_accuracy: f64,
}

/// ロジットと教師値から二値交差エントロピーの平均を求める：softplus(z) − t·z
fn bce_with_logits(logits: Tensor<2>, targets: Tensor<2>) -> Tensor<1> {
    (softplus(logits.clone(), 1.0) - logits * targets).mean()
}

/// サンプルを入力と教師値のテンソルにする。`augment` が真なら局面ごとにランダムな対称変換を適用する。
fn make_batch(samples: &[&Sample], device: &Device, rng: &mut Rng, augment: bool) -> (Tensor<4>, Tensor<2>) {
    let n = samples.len();
    let mut features = vec![0.0f32; n * FEATURE_LEN];
    let mut targets = Vec::with_capacity(n);
    for (i, s) in samples.iter().enumerate() {
        let board = if augment { transform_board(&s.board, rng.below(SYMMETRIES as u32) as usize) } else { s.board };
        write_features(&board, &mut features[i * FEATURE_LEN..(i + 1) * FEATURE_LEN]);
        targets.push(s.target);
    }
    let x = Tensor::<4>::from_data(TensorData::new(features, [n, PLANES, 8, 8]), device);
    let t = Tensor::<2>::from_data(TensorData::new(targets, [n, 1]), device);
    (x, t)
}

fn scalar(t: Tensor<1>) -> f64 {
    t.into_data().try_to_vec::<f32>().expect("f32")[0] as f64
}

/// `model` を `samples` で学習する。`device` は autodiff を有効にしたデバイス。
pub fn train(
    mut model: ValueNet,
    samples: &[Sample],
    config: &TrainConfig,
    device: &Device,
    mut on_epoch: impl FnMut(&EpochReport),
) -> ValueNet {
    let mut rng = Rng::new(config.seed);
    let mut indices: Vec<usize> = (0..samples.len()).collect();
    shuffle(&mut indices, &mut rng);
    let n_val = ((samples.len() as f64 * config.validation_fraction) as usize).min(samples.len() / 2);
    let (val_idx, train_idx) = indices.split_at(n_val);
    let mut train_idx = train_idx.to_vec();

    let mut optim = AdamConfig::new().init();
    for epoch in 1..=config.epochs {
        shuffle(&mut train_idx, &mut rng);
        let (mut loss_sum, mut batches) = (0.0, 0usize);
        for chunk in train_idx.chunks(config.batch_size) {
            let batch: Vec<&Sample> = chunk.iter().map(|&i| &samples[i]).collect();
            let (x, t) = make_batch(&batch, device, &mut rng, true);
            let loss = bce_with_logits(model.forward(x), t);
            loss_sum += scalar(loss.clone().detach());
            batches += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(config.learning_rate, model, grads);
        }

        let (validation_loss, validation_accuracy) = validate(&model, samples, val_idx, device, &mut rng);
        on_epoch(&EpochReport {
            epoch,
            train_loss: loss_sum / batches.max(1) as f64,
            validation_loss,
            validation_accuracy,
        });
    }
    model
}

fn validate(model: &ValueNet, samples: &[Sample], val_idx: &[usize], device: &Device, rng: &mut Rng) -> (f64, f64) {
    if val_idx.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    // 推論モード（バッチ正規化は学習中の移動平均を使い、移動平均を更新しない）で評価する
    let model = model.valid();
    let device = device.clone().without_autodiff();
    let (mut loss_sum, mut correct, mut decided) = (0.0, 0usize, 0usize);
    for chunk in val_idx.chunks(4096) {
        let batch: Vec<&Sample> = chunk.iter().map(|&i| &samples[i]).collect();
        let (x, t) = make_batch(&batch, &device, rng, false);
        let logits = model.forward(x);
        loss_sum += scalar(bce_with_logits(logits.clone(), t)) * batch.len() as f64;
        let logits = logits.into_data().try_to_vec::<f32>().expect("f32");
        for (s, z) in batch.iter().zip(logits) {
            if s.target != 0.5 {
                decided += 1;
                if (z > 0.0) == (s.target > 0.5) {
                    correct += 1;
                }
            }
        }
    }
    (loss_sum / val_idx.len() as f64, correct as f64 / decided.max(1) as f64)
}

fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.below(i as u32 + 1) as usize;
        v.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Color};
    use crate::nn::model::Architecture;

    /// 「手番側の石が多いほど勝ち」という単純なデータを学習できる
    #[test]
    fn learns_simple_pattern() {
        let mut rng = Rng::new(1);
        let mut samples = Vec::new();
        for _ in 0..2000 {
            // ランダムな局面（重ならないように石を置く）
            let (mut black, mut white) = (0u64, 0u64);
            for sq in 0..64 {
                match rng.below(3) {
                    0 => black |= 1 << sq,
                    1 => white |= 1 << sq,
                    _ => {}
                }
            }
            let board = Board::from_bitboards(black, white, Color::Black);
            let target = if board.count(Color::Black) > board.count(Color::White) { 1.0 } else { 0.0 };
            samples.push(Sample { board, target });
        }
        let device = Device::flex().autodiff();
        let config = TrainConfig { epochs: 8, batch_size: 64, validation_fraction: 0.1, ..Default::default() };
        // 小さな構成と、バッチ正規化を含む残差ネットワークの両方で学習できる
        // 少量のデータ・小さなバッチでは、バッチ正規化の移動平均の揺れで検証結果がエポックごとに上下するため、
        // 最終エポックではなく最良のエポックで判定する
        for arch in [Architecture::Small, Architecture::ResNet { channels: 16, blocks: 2 }] {
            device.seed(1);
            let mut reports = Vec::new();
            train(ValueNet::new(arch, &device), &samples, &config, &device, |r| reports.push(*r));
            let (first, last) = (reports.first().unwrap(), reports.last().unwrap());
            assert!(last.train_loss < first.train_loss, "{arch}: {reports:?}");
            let best = reports.iter().map(|r| r.validation_accuracy).fold(0.0, f64::max);
            assert!(best > 0.85, "{arch}: {reports:?}");
        }
    }
}
