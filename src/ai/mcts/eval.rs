//! 葉ノードの評価関数
//!
//! [`Evaluator`] を実装すれば、MCTS の葉ノードの評価方法を置き換えられる
//! （例：ランダムプレイアウト、静的評価関数、ニューラルネット）。

use crate::board::{BitIter, Board, Color, GameResult, Move};
use crate::rng::Rng;

/// 葉ノードの評価関数。探索スレッド間で共有されるため `Send + Sync` が必要。
pub trait Evaluator: Send + Sync {
    /// 表示用の名前
    fn name(&self) -> String;

    /// 終局していない局面 `board` を評価し、**手番側から見た**勝率（0〜1）を返す。
    /// 終局局面は MCTS 側で結果から評価するため、ここには渡されない。
    /// 範囲外の値は 0〜1 に切り詰められる。
    fn evaluate(&self, board: &Board, rng: &mut Rng) -> f64;

    /// 複数の局面をまとめて評価し、それぞれ [`evaluate`](Self::evaluate) と同じ意味の値を同じ順で返す。
    /// 既定は 1 局面ずつ `evaluate` を呼ぶ。まとめて計算すると速い評価関数（NN など）は上書きする。
    fn evaluate_batch(&self, boards: &[Board], rng: &mut Rng) -> Vec<f64> {
        boards.iter().map(|b| self.evaluate(b, rng)).collect()
    }

    /// 1 回の評価で行うプレイアウト回数（統計表示用）。プレイアウトしない評価関数は 0。
    fn playouts_per_eval(&self) -> u32 {
        0
    }

    /// 計算に使うデバイスの表示名（表示用。NN の評価関数なら "GPU（CUDA）" や "CPU"）
    fn device(&self) -> Option<&str> {
        None
    }
}

/// ランダムプレイアウトを `playouts` 回行い、結果（勝ち = 1, 引き分け = 0.5, 負け = 0）の平均を評価値とする
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayoutEvaluator {
    playouts: u32,
}

impl PlayoutEvaluator {
    /// `playouts` は 1 未満なら 1 に丸める
    pub fn new(playouts: u32) -> Self {
        PlayoutEvaluator { playouts: playouts.max(1) }
    }

    pub fn playouts(&self) -> u32 {
        self.playouts
    }
}

impl Default for PlayoutEvaluator {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Evaluator for PlayoutEvaluator {
    fn name(&self) -> String {
        format!("playout x{}", self.playouts)
    }

    fn evaluate(&self, board: &Board, rng: &mut Rng) -> f64 {
        let me = board.turn();
        let total: f64 = (0..self.playouts).map(|_| result_value(playout(*board, rng), me)).sum();
        total / self.playouts as f64
    }

    fn playouts_per_eval(&self) -> u32 {
        self.playouts
    }
}

/// 終局までランダムに打つ
pub fn playout(mut board: Board, rng: &mut Rng) -> GameResult {
    loop {
        let mask = board.legal_mask();
        if mask == 0 {
            match board.play(Move::Pass) {
                Some(next) => board = next,
                None => return board.result().expect("game over"),
            }
            continue;
        }
        let k = rng.below(mask.count_ones()) as usize;
        let sq = BitIter(mask).nth(k).unwrap();
        board = board.play(Move::Place(sq)).expect("legal move");
    }
}

/// 対局結果を `perspective` 側から見た評価値（勝ち = 1, 引き分け = 0.5, 負け = 0）にする
pub fn result_value(result: GameResult, perspective: Color) -> f64 {
    match result {
        GameResult::Win(c) if c == perspective => 1.0,
        GameResult::Win(_) => 0.0,
        GameResult::Draw => 0.5,
    }
}

/// マスの重み（隅を高く、隅の隣を低く評価する定番の表）
#[rustfmt::skip]
const SQUARE_WEIGHTS: [i32; 64] = [
    100, -20,  10,   5,   5,  10, -20, 100,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
    100, -20,  10,   5,   5,  10, -20, 100,
];

/// 静的評価関数：マスの重みと着手可能数（mobility）の差をシグモイドで勝率に変換する。
/// プレイアウトしないので 1 回の評価が非常に速い（評価関数の置き換え例）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionalEvaluator {
    /// 着手可能数の差 1 あたりの点数
    pub mobility_weight: f64,
    /// 点数を勝率に変換するときの尺度（大きいほど 0.5 寄りになる）
    pub scale: f64,
}

impl Default for PositionalEvaluator {
    fn default() -> Self {
        PositionalEvaluator { mobility_weight: 5.0, scale: 40.0 }
    }
}

impl Evaluator for PositionalEvaluator {
    fn name(&self) -> String {
        "positional".to_string()
    }

    fn evaluate(&self, board: &Board, _rng: &mut Rng) -> f64 {
        let me = board.turn();
        let weight_sum = |bits: u64| BitIter(bits).map(|sq| SQUARE_WEIGHTS[sq as usize]).sum::<i32>();
        let positional = weight_sum(board.stones(me)) - weight_sum(board.stones(me.opponent()));

        let opponent_view = board.with_turn(me.opponent());
        let mobility = board.legal_mask().count_ones() as i32 - opponent_view.legal_mask().count_ones() as i32;

        let score = positional as f64 + self.mobility_weight * mobility as f64;
        1.0 / (1.0 + (-score / self.scale).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positional_evaluator_is_symmetric_at_start() {
        let v = PositionalEvaluator::default().evaluate(&Board::new(), &mut Rng::new(0));
        assert!((v - 0.5).abs() < 1e-9, "{v}");
    }

    #[test]
    fn positional_evaluator_prefers_corners() {
        // 黒が隅 a1 を持つ局面（白番）は、手番の白から見て不利
        let board = Board::from_bitboards(1 | (1 << 28) | (1 << 35), (1 << 27) | (1 << 36), Color::White);
        let v = PositionalEvaluator::default().evaluate(&board, &mut Rng::new(0));
        assert!(v < 0.2, "{v}");
    }

    #[test]
    fn playout_evaluator_returns_side_to_move_perspective() {
        // 白番で白が圧倒的に多い局面 → 白（手番側）から見て勝率が高い
        let white = !((1 << 63) | (1 << 62) | 1);
        let board = Board::from_bitboards(1, white, Color::White);
        let v = PlayoutEvaluator::new(20).evaluate(&board, &mut Rng::new(0));
        assert!(v > 0.9, "{v}");
    }
}
