//! 思考アルゴリズム。`Ai` トレイトを実装することで切り替え可能にする。

pub mod mcts;
pub mod random;

use std::sync::atomic::AtomicBool;

use crate::board::{Board, Move};

pub use mcts::{
    Evaluator, Mcts, MctsConfig, ParallelMode, PlayoutEvaluator, PositionalEvaluator, SearchLimit, result_value,
};
pub use random::RandomAi;

pub trait Ai {
    fn name(&self) -> String;

    /// 手番側の着手を選ぶ。終局盤面では `None`。
    fn select_move(&mut self, board: &Board) -> Option<Move>;

    /// 直前の思考内容の要約（GUI 表示用）。表示するものがなければ `None`。
    fn summary(&self) -> Option<String> {
        None
    }

    /// 相手の手番中に探索を進める（ponder）に対応しているか
    fn can_ponder(&self) -> bool {
        false
    }

    /// 相手の手番の局面 `board` で、`stop` が立つまで探索を進める。
    /// 結果は次の `select_move` で活用される。対応しない AI は何もしない。
    fn ponder(&mut self, _board: &Board, _stop: &AtomicBool) {}
}
