//! 自己対戦による学習データの生成と、評価関数どうしの対戦
//!
//! 多数の対局を並行して進める（各対局は 1 スレッドの MCTS）。評価関数がバッチ推論なら、
//! 並行対局からの評価要求がまとまって GPU で処理される。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use super::data::Sample;
use crate::ai::{Evaluator, Mcts, MctsConfig, SearchLimit, result_value};
use crate::board::{Board, Color, GameResult, Move};
use crate::rng::Rng;

#[derive(Debug, Clone, Copy)]
pub struct SelfPlayConfig {
    pub games: usize,
    /// 同時に進める対局数
    pub concurrency: usize,
    /// 1 手あたりの MCTS の反復回数
    pub iterations: u32,
    /// 序盤のこの手数までは、訪問回数に比例した確率で手を選ぶ（対局の多様性のため）
    pub sampling_moves: usize,
    pub seed: u64,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig { games: 1000, concurrency: 256, iterations: 200, sampling_moves: 8, seed: 0 }
    }
}

fn mcts_config(iterations: u32) -> MctsConfig {
    MctsConfig {
        limit: SearchLimit::Iterations(iterations),
        threads: 1,
        // 反復回数が少ないので上限は小さくてよい（並行対局数ぶん確保されるため）
        max_nodes: 200_000,
        ..MctsConfig::default()
    }
}

/// MCTS で手を選ぶ。`sample` が真なら訪問回数に比例した確率で選ぶ。
fn choose_move(mcts: &mut Mcts, board: &Board, sample: bool, rng: &mut Rng) -> Move {
    let best = mcts.search(board).expect("game is not over");
    let children = &mcts.last_stats().children;
    if !sample || children.is_empty() {
        return best;
    }
    let total: u64 = children.iter().map(|c| c.1 as u64).sum();
    if total == 0 {
        return best;
    }
    let mut r = (rng.next_u64() % total) as i64;
    for &(mv, visits, _) in children {
        r -= visits as i64;
        if r < 0 {
            return mv;
        }
    }
    best
}

/// 1 局の自己対戦。各局面と、その手番側から見た最終結果を返す。
fn play_self_game(evaluator: &Arc<dyn Evaluator>, config: &SelfPlayConfig, seed: u64) -> (Vec<Sample>, GameResult) {
    let mut rng = Rng::new(seed);
    let mut mcts = Mcts::with_evaluator(mcts_config(config.iterations), evaluator.clone());
    mcts.set_seed(rng.next_u64());

    let mut board = Board::new();
    let mut positions = Vec::new();
    let mut ply = 0;
    while !board.is_game_over() {
        positions.push(board);
        let mv = choose_move(&mut mcts, &board, ply < config.sampling_moves, &mut rng);
        board = board.play(mv).expect("legal move");
        ply += 1;
    }
    let result = board.result().expect("game over");
    let samples = positions
        .into_iter()
        .map(|b| Sample { board: b, target: result_value(result, b.turn()) as f32 })
        .collect();
    (samples, result)
}

/// 自己対戦で学習データを生成する。`progress` は 1 局終わるごとに完了局数で呼ばれる。
pub fn self_play(
    evaluator: Arc<dyn Evaluator>,
    config: &SelfPlayConfig,
    progress: impl Fn(usize) + Sync,
) -> Vec<Sample> {
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let samples = Mutex::new(Vec::new());
    thread::scope(|scope| {
        for _ in 0..config.concurrency.clamp(1, config.games.max(1)) {
            scope.spawn(|| {
                loop {
                    let game = next.fetch_add(1, Ordering::Relaxed);
                    if game >= config.games {
                        break;
                    }
                    let seed = config.seed.wrapping_mul(1_000_003).wrapping_add(game as u64);
                    let (game_samples, _) = play_self_game(&evaluator, config, seed);
                    samples.lock().unwrap().extend(game_samples);
                    progress(done.fetch_add(1, Ordering::Relaxed) + 1);
                }
            });
        }
    });
    samples.into_inner().unwrap()
}

/// 対戦成績（A から見た勝ち・負け・引き分け）
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MatchResult {
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
}

impl MatchResult {
    /// 勝率（引き分けは 0.5 勝）
    pub fn score(&self) -> f64 {
        let games = self.wins + self.losses + self.draws;
        (self.wins as f64 + 0.5 * self.draws as f64) / games.max(1) as f64
    }
}

/// 評価関数 A と B を、同じ反復回数の MCTS で対戦させる。先後は交互。
/// 序盤 `sampling_moves` 手は訪問回数に比例して手を選び、毎局違う展開にする。
pub fn play_match(
    a: Arc<dyn Evaluator>,
    b: Arc<dyn Evaluator>,
    config: &SelfPlayConfig,
    progress: impl Fn(usize) + Sync,
) -> MatchResult {
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let result = Mutex::new(MatchResult::default());
    thread::scope(|scope| {
        for _ in 0..config.concurrency.clamp(1, config.games.max(1)) {
            scope.spawn(|| {
                loop {
                    let game = next.fetch_add(1, Ordering::Relaxed);
                    if game >= config.games {
                        break;
                    }
                    let mut rng = Rng::new(config.seed.wrapping_mul(1_000_003).wrapping_add(game as u64));
                    let a_color = if game.is_multiple_of(2) { Color::Black } else { Color::White };
                    let mut players = [
                        Mcts::with_evaluator(mcts_config(config.iterations), a.clone()),
                        Mcts::with_evaluator(mcts_config(config.iterations), b.clone()),
                    ];
                    for p in &mut players {
                        p.set_seed(rng.next_u64());
                    }
                    let mut board = Board::new();
                    let mut ply = 0;
                    while !board.is_game_over() {
                        let idx = usize::from(board.turn() != a_color);
                        let mv = choose_move(&mut players[idx], &board, ply < config.sampling_moves, &mut rng);
                        board = board.play(mv).expect("legal move");
                        ply += 1;
                    }
                    let mut r = result.lock().unwrap();
                    match board.result().expect("game over") {
                        GameResult::Win(c) if c == a_color => r.wins += 1,
                        GameResult::Win(_) => r.losses += 1,
                        GameResult::Draw => r.draws += 1,
                    }
                    drop(r);
                    progress(done.fetch_add(1, Ordering::Relaxed) + 1);
                }
            });
        }
    });
    result.into_inner().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{PlayoutEvaluator, PositionalEvaluator};

    #[test]
    fn self_play_labels_positions_from_side_to_move() {
        let config = SelfPlayConfig { games: 4, concurrency: 4, iterations: 30, ..Default::default() };
        let samples = self_play(Arc::new(PlayoutEvaluator::new(1)), &config, |_| {});
        assert!(samples.len() >= 4 * 9);
        assert!(samples.iter().all(|s| [0.0, 0.5, 1.0].contains(&s.target) && !s.board.is_game_over()));
        // 同じ局（初期局面から始まる）の黒番と白番は逆の教師値を持つ
        // 1 局分のサンプルは連続して追加されるので、初期局面の位置で区切れる
        let starts: Vec<usize> = (0..samples.len()).filter(|&i| samples[i].board == Board::new()).collect();
        assert_eq!(starts.len(), 4);
        let first = &samples[starts[0]..starts[1]];
        let black = first.iter().find(|s| s.board.turn() == Color::Black).unwrap().target;
        let white = first.iter().find(|s| s.board.turn() == Color::White).unwrap().target;
        assert_eq!(black + white, 1.0);
    }

    #[test]
    fn match_counts_all_games() {
        let config = SelfPlayConfig { games: 6, concurrency: 3, iterations: 30, ..Default::default() };
        let r = play_match(Arc::new(PositionalEvaluator::default()), Arc::new(PlayoutEvaluator::new(1)), &config, |_| {});
        assert_eq!(r.wins + r.losses + r.draws, 6);
    }
}
