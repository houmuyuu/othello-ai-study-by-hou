use super::Ai;
use crate::board::{Board, Move};
use crate::rng::Rng;

/// 合法手から一様ランダムに選ぶ AI（比較用）
pub struct RandomAi {
    rng: Rng,
}

impl RandomAi {
    pub fn new() -> Self {
        RandomAi { rng: Rng::from_time() }
    }

    pub fn with_seed(seed: u64) -> Self {
        RandomAi { rng: Rng::new(seed) }
    }
}

impl Default for RandomAi {
    fn default() -> Self {
        Self::new()
    }
}

impl Ai for RandomAi {
    fn name(&self) -> String {
        "Random".to_string()
    }

    fn select_move(&mut self, board: &Board) -> Option<Move> {
        let moves = board.legal_moves();
        if moves.is_empty() {
            return None;
        }
        Some(moves[self.rng.below(moves.len() as u32) as usize])
    }
}
