//! ニューラルネットによる評価関数（価値ネットワーク）と、その強化学習
//!
//! 自己対戦の最終結果を教師データとして価値ネットワークを学習し、MCTS の評価関数として使う。

pub mod data;
pub mod device;
pub mod evaluator;
pub mod features;
pub mod model;
pub mod selfplay;
pub mod train;

pub use device::{InferenceDevice, inference_device, inference_device_if_ready, use_bundled_libdevice};
pub use evaluator::{BatchedNnEvaluator, NnEvaluator, load_evaluator};
pub use model::{Architecture, ValueNet};
