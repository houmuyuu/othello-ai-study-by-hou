//! モンテカルロ木探索（UCT）
//!
//! 葉ノードの評価は [`Evaluator`] で置き換えられる。既定は [`PlayoutEvaluator`]
//! （ランダムプレイアウトを指定回数行い、結果（勝ち = 1, 引き分け = 0.5, 負け = 0）を平均した値）。
//! 終局局面は評価関数を使わず、結果から正確に評価する。
//!
//! マルチスレッド時の並列化方式は 2 種類から選べる（[`ParallelMode`]）。
//!
//! 探索木は探索後も保持し（`reuse_tree`）、次の探索局面が木の中にあればその部分木から探索を続ける。
//! 相手の手番中に探索を進める ponder（[`Ai::ponder`]）も、この再利用によって次の探索に活かされる。

mod eval;
mod root;
mod tree;

pub use eval::{Evaluator, PlayoutEvaluator, PositionalEvaluator, playout, result_value};

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use super::Ai;
use crate::board::{Board, Color, Move};
use crate::rng::Rng;

/// 1 手あたりの探索量
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchLimit {
    /// 探索の反復回数（選択→展開→評価→逆伝播 を 1 回とする）。全スレッドの合計。
    Iterations(u32),
    /// 思考時間
    Time(Duration),
}

/// 並列化方式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelMode {
    /// 木の並列化：全スレッドで 1 本の木を共有し、バーチャルロスで探索先を分散させる
    Tree,
    /// ルート並列化：スレッドごとに独立した木で探索し、終了時にルートの統計を合算する
    Root,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MctsConfig {
    /// 1 手あたりの探索量
    pub limit: SearchLimit,
    /// UCB1 の探索定数
    pub exploration: f64,
    /// 探索スレッド数（1 でシングルスレッド）
    pub threads: u32,
    /// マルチスレッド時の並列化方式
    pub parallel: ParallelMode,
    /// ノード数の上限（全スレッド合計）。達したら展開をやめ、既存の葉からのプレイアウトだけを続ける。
    pub max_nodes: usize,
    /// 探索木を次の探索に再利用するか
    pub reuse_tree: bool,
    /// 1 スレッドが 1 回の評価にまとめる葉の数（1 で 1 葉ずつ評価）。木の並列化のときだけ有効。
    ///
    /// バーチャルロスを残したまま選択を繰り返して葉を集め、[`Evaluator::evaluate_batch`] でまとめて評価する。
    /// まとめて計算すると速い評価関数（NN）で探索が速くなるが、古い統計で葉を選ぶ分だけ探索の質は下がる。
    pub batch_size: u32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            limit: SearchLimit::Time(Duration::from_secs(5)),
            exploration: std::f64::consts::SQRT_2,
            threads: 1,
            parallel: ParallelMode::Tree,
            max_nodes: 5_000_000,
            reuse_tree: true,
            batch_size: 1,
        }
    }
}

/// 直前の探索の統計（GUI での表示やデバッグ用）
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub iterations: u32,
    pub playouts: u64,
    /// 探索後のノード数（全スレッド合計）
    pub nodes: usize,
    /// 前回までの探索木から引き継いだノード数
    pub reused_nodes: usize,
    pub elapsed: Duration,
    /// (手, 訪問回数, 平均評価値) を訪問回数の降順で
    pub children: Vec<(Move, u32, f64)>,
}

/// 各並列化方式の探索結果
struct SearchOutput {
    iterations: u32,
    playouts: u64,
    nodes: usize,
    /// ルートの子の (手, 訪問回数, 評価値合計)。同じ手が複数回現れてもよい（合算される）。
    root_children: Vec<(Move, u32, f64)>,
}

/// 保持している探索木（並列化方式ごとに形が異なる）
enum SearchTree {
    Tree(tree::Tree),
    Root(root::Forest),
}

impl SearchTree {
    fn nodes(&self) -> usize {
        match self {
            SearchTree::Tree(t) => t.nodes(),
            SearchTree::Root(f) => f.nodes(),
        }
    }
}

/// 木の再利用で、前回のルートから何手先までの局面を探すか
const MAX_REUSE_DEPTH: u32 = 4;

/// ルートの訪問回数の上限（u32 のオーバーフローを避けるため、長時間の ponder でもここで止める）
const MAX_ROOT_VISITS: u32 = 1 << 31;

pub struct Mcts {
    config: MctsConfig,
    evaluator: Arc<dyn Evaluator>,
    rng: Rng,
    last_stats: SearchStats,
    tree: Option<SearchTree>,
}

impl Mcts {
    /// 評価関数は既定の [`PlayoutEvaluator`]（プレイアウト 1 回）
    pub fn new(config: MctsConfig) -> Self {
        Self::with_evaluator(config, Arc::new(PlayoutEvaluator::default()))
    }

    pub fn with_evaluator(config: MctsConfig, evaluator: Arc<dyn Evaluator>) -> Self {
        Mcts { config, evaluator, rng: Rng::from_time(), last_stats: SearchStats::default(), tree: None }
    }

    /// 乱数の種を固定して作る（テストや再現用）。評価関数は既定。
    pub fn with_seed(config: MctsConfig, seed: u64) -> Self {
        let mut mcts = Self::new(config);
        mcts.set_seed(seed);
        mcts
    }

    /// 乱数の種を固定する（テストや再現用）
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    pub fn config(&self) -> &MctsConfig {
        &self.config
    }

    pub fn evaluator(&self) -> &dyn Evaluator {
        &*self.evaluator
    }

    /// 評価関数を置き換える。評価値の基準が変わるため、保持している探索木は破棄する。
    pub fn set_evaluator(&mut self, evaluator: Arc<dyn Evaluator>) {
        self.evaluator = evaluator;
        self.tree = None;
    }

    pub fn set_limit(&mut self, limit: SearchLimit) {
        self.config.limit = limit;
    }

    pub fn set_exploration(&mut self, c: f64) {
        self.config.exploration = c;
    }

    /// 探索スレッド数を変更する（1 未満は 1 に丸める）
    pub fn set_threads(&mut self, n: u32) {
        self.config.threads = n.max(1);
    }

    /// 1 回の評価にまとめる葉の数を変更する（1 未満は 1 に丸める）
    pub fn set_batch_size(&mut self, n: u32) {
        self.config.batch_size = n.max(1);
    }

    pub fn set_parallel_mode(&mut self, mode: ParallelMode) {
        self.config.parallel = mode;
    }

    pub fn set_max_nodes(&mut self, n: usize) {
        self.config.max_nodes = n;
    }

    /// 木の再利用の有無を変更する。無効にすると保持している木は破棄する。
    pub fn set_reuse_tree(&mut self, reuse: bool) {
        self.config.reuse_tree = reuse;
        if !reuse {
            self.tree = None;
        }
    }

    /// 保持している探索木を破棄する（新しい対局を始めるときなど）
    pub fn clear_tree(&mut self) {
        self.tree = None;
    }

    pub fn last_stats(&self) -> &SearchStats {
        &self.last_stats
    }

    /// 探索して最善手（訪問回数最大の手）を返す。終局盤面では `None`。
    pub fn search(&mut self, root_board: &Board) -> Option<Move> {
        let moves = root_board.legal_moves();
        match moves.len() {
            0 => return None,
            1 => {
                self.last_stats = SearchStats::default();
                return Some(moves[0]);
            }
            _ => {}
        }

        self.last_stats = self.run_search(root_board, self.config, None);
        self.last_stats.children.first().map(|c| c.0).or_else(|| moves.first().copied())
    }

    /// 相手の手番中に `stop` が立つまで探索を進める。結果は木に蓄積され、次の探索で再利用される。
    /// 木の再利用が無効なら何もしない。
    pub fn ponder(&mut self, board: &Board, stop: &AtomicBool) {
        if !self.config.reuse_tree || board.is_game_over() {
            return;
        }
        let config = MctsConfig { limit: SearchLimit::Time(Duration::MAX), ..self.config };
        self.run_search(board, config, Some(stop));
    }

    /// 保持している木から `board` に該当する部分木を取り出す。なければ新しい木を作る。
    /// 戻り値の 2 つ目は引き継いだノード数（新しい木なら 0）。
    fn take_tree(&mut self, board: &Board) -> (SearchTree, usize) {
        let reused = match (self.tree.take(), self.config.parallel) {
            (Some(SearchTree::Tree(t)), ParallelMode::Tree) if self.config.reuse_tree => {
                t.reroot(board).map(SearchTree::Tree)
            }
            (Some(SearchTree::Root(f)), ParallelMode::Root) if self.config.reuse_tree => {
                f.reroot(board).map(SearchTree::Root)
            }
            _ => None,
        };
        match reused {
            Some(t) => {
                let nodes = t.nodes();
                (t, nodes)
            }
            None => match self.config.parallel {
                ParallelMode::Tree => (SearchTree::Tree(tree::Tree::new(*board)), 0),
                ParallelMode::Root => (SearchTree::Root(root::Forest::new(*board)), 0),
            },
        }
    }

    fn run_search(&mut self, root_board: &Board, config: MctsConfig, stop: Option<&AtomicBool>) -> SearchStats {
        let start = Instant::now();
        let (mut search_tree, reused_nodes) = self.take_tree(root_board);
        let output = match &mut search_tree {
            SearchTree::Tree(t) => t.search(&config, &*self.evaluator, &mut self.rng, start, stop),
            SearchTree::Root(f) => f.search(&config, &*self.evaluator, &mut self.rng, start, stop),
        };
        if config.reuse_tree {
            self.tree = Some(search_tree);
        }

        let mut children: Vec<(Move, u32, f64)> = Vec::new();
        for (mv, visits, value_sum) in output.root_children {
            match children.iter_mut().find(|c| c.0 == mv) {
                Some(c) => {
                    c.1 += visits;
                    c.2 += value_sum;
                }
                None => children.push((mv, visits, value_sum)),
            }
        }
        for c in &mut children {
            c.2 /= c.1.max(1) as f64;
        }
        children.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.total_cmp(&a.2)));

        SearchStats {
            iterations: output.iterations,
            playouts: output.playouts,
            nodes: output.nodes,
            reused_nodes,
            elapsed: start.elapsed(),
            children,
        }
    }
}

/// 時間制限の確認間隔（時間計測のコストを抑えるため毎回は確認しない）
const TIME_CHECK_INTERVAL: u32 = 64;

/// 葉ノードを評価し、黒視点の評価値（0〜1）と実行したプレイアウト回数を返す。
/// 終局局面は結果が確定しているので評価関数を使わない。
fn evaluate(board: &Board, evaluator: &dyn Evaluator, rng: &mut Rng) -> (f64, u32) {
    if let Some(result) = board.result() {
        return (result_value(result, Color::Black), 0);
    }
    let value = evaluator.evaluate(board, rng);
    (black_value(board, value), evaluator.playouts_per_eval())
}

/// 評価関数が返した `board` の手番側視点の値を、黒視点の評価値（0〜1）にする
fn black_value(board: &Board, value: f64) -> f64 {
    // 評価関数の誤りで NaN 等が返っても探索が壊れないよう、引き分け扱い・範囲内に収める
    let value = if value.is_finite() { value.clamp(0.0, 1.0) } else { 0.5 };
    value_for(board.turn(), value)
}

/// 黒視点の評価値を、`mover` 視点に変換する
fn value_for(mover: Color, value_black: f64) -> f64 {
    match mover {
        Color::Black => value_black,
        Color::White => 1.0 - value_black,
    }
}

impl Ai for Mcts {
    fn name(&self) -> String {
        let limit = match self.config.limit {
            SearchLimit::Iterations(n) => format!("{n} iter"),
            SearchLimit::Time(t) => format!("{:.1}s", t.as_secs_f64()),
        };
        let batch = if self.config.batch_size > 1 { format!(", batch {}", self.config.batch_size) } else { String::new() };
        format!(
            "MCTS({limit}, {}, {} threads, {:?}{batch})",
            self.evaluator.name(),
            self.config.threads,
            self.config.parallel
        )
    }

    fn select_move(&mut self, board: &Board) -> Option<Move> {
        self.search(board)
    }

    fn can_ponder(&self) -> bool {
        self.config.reuse_tree
    }

    fn ponder(&mut self, board: &Board, stop: &AtomicBool) {
        Mcts::ponder(self, board, stop);
    }

    fn summary(&self) -> Option<String> {
        let s = &self.last_stats;
        let Some(&(mv, visits, value)) = s.children.first() else {
            return Some("合法手が 1 つのため探索なし".to_string());
        };
        let mut text = format!(
            "{mv}: 勝率 {:.1}% (訪問 {visits})\n反復 {} / プレイアウト {} / {:.2} 秒\nノード {} (うち再利用 {})",
            value * 100.0,
            s.iterations,
            s.playouts,
            s.elapsed.as_secs_f64(),
            s.nodes,
            s.reused_nodes
        );
        if let Some(device) = self.evaluator.device() {
            text += &format!("\n推論デバイス {device}");
        }
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    use super::*;
    use crate::ai::RandomAi;
    use crate::board::GameResult;

    fn mcts_with(config: MctsConfig, evaluator: impl Evaluator + 'static, seed: u64) -> Mcts {
        let mut mcts = Mcts::with_evaluator(config, Arc::new(evaluator));
        mcts.set_seed(seed);
        mcts
    }

    const MODES: [ParallelMode; 2] = [ParallelMode::Tree, ParallelMode::Root];

    fn play_game(black: &mut dyn Ai, white: &mut dyn Ai) -> GameResult {
        let mut board = Board::new();
        while !board.legal_moves().is_empty() {
            let ai: &mut dyn Ai = if board.turn() == Color::Black { black } else { white };
            let mv = ai.select_move(&board).unwrap();
            board = board.play(mv).expect("AI returned an illegal move");
        }
        board.result().unwrap()
    }

    #[test]
    fn playouts_per_eval_is_reflected_in_stats() {
        for parallel in MODES {
            let config = MctsConfig { limit: SearchLimit::Iterations(50), parallel, ..Default::default() };
            let mut mcts = mcts_with(config, PlayoutEvaluator::new(7), 1);
            mcts.search(&Board::new()).unwrap();
            assert_eq!(mcts.last_stats().iterations, 50);
            assert_eq!(mcts.last_stats().playouts, 350);

            // 後から回数を変更できる
            mcts.set_evaluator(Arc::new(PlayoutEvaluator::new(3)));
            mcts.search(&Board::new()).unwrap();
            assert_eq!(mcts.last_stats().playouts, 150);
        }
    }

    #[test]
    fn multithreaded_search_counts_iterations_exactly() {
        for parallel in MODES {
            let config =
                MctsConfig { limit: SearchLimit::Iterations(1001), threads: 4, parallel, ..Default::default() };
            let mut mcts = mcts_with(config, PlayoutEvaluator::new(2), 1);
            mcts.search(&Board::new()).unwrap();
            let stats = mcts.last_stats();
            assert_eq!(stats.iterations, 1001, "{parallel:?}");
            assert_eq!(stats.playouts, 2002, "{parallel:?}");
            // 各反復はルートの子をちょうど 1 つ通る
            assert_eq!(stats.children.iter().map(|c| c.1).sum::<u32>(), 1001, "{parallel:?}");
            assert_eq!(stats.children.len(), 4, "{parallel:?}");
        }
    }

    #[test]
    fn multithreaded_time_limited_search_runs_more_iterations() {
        for parallel in MODES {
            let limit = SearchLimit::Time(Duration::from_millis(200));
            let config = MctsConfig { limit, parallel, ..Default::default() };
            let mut single = Mcts::with_seed(MctsConfig { threads: 1, ..config }, 1);
            let mut multi = Mcts::with_seed(MctsConfig { threads: 4, ..config }, 1);
            single.search(&Board::new()).unwrap();
            multi.search(&Board::new()).unwrap();
            let (s, m) = (single.last_stats().iterations, multi.last_stats().iterations);
            assert!(m > s * 2, "{parallel:?} single: {s}, multi: {m}");
        }
    }

    #[test]
    fn node_limit_is_respected() {
        for parallel in MODES {
            let config = MctsConfig {
                limit: SearchLimit::Iterations(5000),
                threads: 4,
                parallel,
                max_nodes: 200,
                ..Default::default()
            };
            let mut mcts = Mcts::with_seed(config, 1);
            mcts.search(&Board::new()).unwrap();
            let stats = mcts.last_stats();
            assert!(stats.nodes <= 200, "{parallel:?}: {} nodes", stats.nodes);
            // 上限に達した後も探索（プレイアウト）は続く
            assert_eq!(stats.iterations, 5000, "{parallel:?}");
        }
    }

    /// 探索→自分の着手→相手の着手 の後の局面で、木が再利用されることを確認する
    #[test]
    fn tree_is_reused_after_two_moves() {
        for parallel in MODES {
            let config =
                MctsConfig { limit: SearchLimit::Iterations(3000), threads: 2, parallel, ..Default::default() };
            let mut mcts = Mcts::with_seed(config, 1);
            let board = Board::new();
            let mv = mcts.search(&board).unwrap();
            let after_own = board.play(mv).unwrap();
            let reply = after_own.legal_moves()[0];
            let next = after_own.play(reply).unwrap();

            mcts.search(&next).unwrap();
            let stats = mcts.last_stats();
            assert!(stats.reused_nodes > 1, "{parallel:?}: reused {}", stats.reused_nodes);
            // 引き継いだ訪問回数の分、ルートの訪問回数の合計は今回の反復回数より多い
            let visits: u32 = stats.children.iter().map(|c| c.1).sum();
            assert!(visits > 3000, "{parallel:?}: visits {visits}");
        }
    }

    #[test]
    fn tree_is_not_reused_when_disabled() {
        for parallel in MODES {
            let config = MctsConfig {
                limit: SearchLimit::Iterations(1000),
                parallel,
                reuse_tree: false,
                ..Default::default()
            };
            let mut mcts = Mcts::with_seed(config, 1);
            mcts.search(&Board::new()).unwrap();
            mcts.search(&Board::new()).unwrap();
            assert_eq!(mcts.last_stats().reused_nodes, 0, "{parallel:?}");
            assert_eq!(mcts.last_stats().children.iter().map(|c| c.1).sum::<u32>(), 1000);
        }
    }

    #[test]
    fn ponder_result_is_used_by_next_search() {
        for parallel in MODES {
            let config =
                MctsConfig { limit: SearchLimit::Iterations(100), threads: 2, parallel, ..Default::default() };
            let mut mcts = Mcts::with_seed(config, 1);
            // 相手（黒）の手番の局面で ponder する
            let board = Board::new();
            let stop = AtomicBool::new(false);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    std::thread::sleep(Duration::from_millis(100));
                    stop.store(true, std::sync::atomic::Ordering::Relaxed);
                });
                mcts.ponder(&board, &stop);
            });
            // 相手が指した後の局面で探索すると、ponder 中に育てた部分木が使われる
            let next = board.play(Move::Place(19)).unwrap(); // d3
            mcts.search(&next).unwrap();
            let stats = mcts.last_stats();
            assert!(stats.reused_nodes > 100, "{parallel:?}: reused {}", stats.reused_nodes);
            assert!(stats.children.iter().map(|c| c.1).sum::<u32>() > 100, "{parallel:?}");
        }
    }

    /// 評価関数の呼び出し回数を数える（評価値は常に 0.5）
    struct CountingEvaluator(AtomicU64);

    impl Evaluator for CountingEvaluator {
        fn name(&self) -> String {
            "counting".to_string()
        }

        fn evaluate(&self, _board: &Board, _rng: &mut Rng) -> f64 {
            self.0.fetch_add(1, Ordering::Relaxed);
            0.5
        }
    }

    #[test]
    fn custom_evaluator_is_called_once_per_iteration() {
        for parallel in MODES {
            let evaluator = Arc::new(CountingEvaluator(AtomicU64::new(0)));
            let config =
                MctsConfig { limit: SearchLimit::Iterations(500), threads: 4, parallel, ..Default::default() };
            let mut mcts = Mcts::with_evaluator(config, evaluator.clone());
            mcts.search(&Board::new()).unwrap();
            // 序盤 500 回の反復では終局局面に届かないので、全反復で評価関数が呼ばれる
            assert_eq!(evaluator.0.load(Ordering::Relaxed), 500, "{parallel:?}");
            assert_eq!(mcts.last_stats().playouts, 0, "{parallel:?}");
        }
    }

    /// 評価関数の戻り値が手番側視点として扱われることを確認する。
    /// 「手番側の石の割合」を返す評価関数では、黒が 1 手指した後（白番）の評価は白の割合（小さい）になる。
    /// 正しく黒視点に変換されていれば、黒から見た勝率は 0.5 を超える。
    #[test]
    fn evaluator_value_is_from_side_to_move() {
        struct Greedy;
        impl Evaluator for Greedy {
            fn name(&self) -> String {
                "greedy".to_string()
            }
            fn evaluate(&self, board: &Board, _rng: &mut Rng) -> f64 {
                let me = board.turn();
                let (mine, theirs) = (board.count(me) as f64, board.count(me.opponent()) as f64);
                mine / (mine + theirs)
            }
        }
        let config = MctsConfig { limit: SearchLimit::Iterations(200), ..Default::default() };
        let mut mcts = mcts_with(config, Greedy, 1);
        mcts.search(&Board::new()).unwrap();
        let value = mcts.last_stats().children[0].2;
        assert!(value > 0.5, "{value}");
    }

    #[test]
    fn invalid_evaluator_values_do_not_break_search() {
        struct Broken;
        impl Evaluator for Broken {
            fn name(&self) -> String {
                "broken".to_string()
            }
            fn evaluate(&self, board: &Board, _rng: &mut Rng) -> f64 {
                if board.count(Color::Black).is_multiple_of(2) { f64::NAN } else { 5.0 }
            }
        }
        let config = MctsConfig { limit: SearchLimit::Iterations(300), threads: 2, ..Default::default() };
        let mut mcts = mcts_with(config, Broken, 1);
        mcts.search(&Board::new()).unwrap();
        assert!(mcts.last_stats().children.iter().all(|c| (0.0..=1.0).contains(&c.2)));
    }

    /// まとめて評価された局面数と、1 回にまとめられた局面数の最大を記録する（評価値は常に 0.5）
    #[derive(Default)]
    struct BatchRecorder {
        evaluated: AtomicU64,
        calls: AtomicU64,
        max_batch: AtomicU64,
    }

    impl Evaluator for BatchRecorder {
        fn name(&self) -> String {
            "batch recorder".to_string()
        }

        fn evaluate(&self, _board: &Board, _rng: &mut Rng) -> f64 {
            unreachable!("batched search uses evaluate_batch")
        }

        fn evaluate_batch(&self, boards: &[Board], _rng: &mut Rng) -> Vec<f64> {
            self.evaluated.fetch_add(boards.len() as u64, Ordering::Relaxed);
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.max_batch.fetch_max(boards.len() as u64, Ordering::Relaxed);
            vec![0.5; boards.len()]
        }
    }

    #[test]
    fn batched_search_evaluates_leaves_together() {
        for threads in [1, 4] {
            let evaluator = Arc::new(BatchRecorder::default());
            let config = MctsConfig {
                limit: SearchLimit::Iterations(1000),
                threads,
                batch_size: 16,
                ..Default::default()
            };
            let mut mcts = Mcts::with_evaluator(config, evaluator.clone());
            mcts.search(&Board::new()).unwrap();
            let stats = mcts.last_stats();
            // 反復回数は指定どおりで、各反復はルートの子をちょうど 1 つ通る
            assert_eq!(stats.iterations, 1000, "x{threads}");
            assert_eq!(stats.children.iter().map(|c| c.1).sum::<u32>(), 1000, "x{threads}");
            // 序盤 1000 回の反復では終局局面に届かないので、全反復の葉がまとめて評価される
            assert_eq!(evaluator.evaluated.load(Ordering::Relaxed), 1000, "x{threads}");
            assert_eq!(evaluator.max_batch.load(Ordering::Relaxed), 16, "x{threads}");
            assert!(evaluator.calls.load(Ordering::Relaxed) <= 1000 / 16 + threads as u64, "x{threads}");
        }
    }

    #[test]
    fn batched_search_counts_uneven_iterations_exactly() {
        let config = MctsConfig {
            limit: SearchLimit::Iterations(1001),
            threads: 4,
            batch_size: 7,
            ..Default::default()
        };
        let mut mcts = mcts_with(config, PlayoutEvaluator::new(2), 1);
        mcts.search(&Board::new()).unwrap();
        let stats = mcts.last_stats();
        assert_eq!(stats.iterations, 1001);
        assert_eq!(stats.playouts, 2002);
        assert_eq!(stats.children.iter().map(|c| c.1).sum::<u32>(), 1001);
    }

    /// 終局局面が混ざったバッチでも、終局局面は評価関数に渡されず結果から評価される
    #[test]
    fn batched_search_handles_terminal_leaves() {
        // ランダムに打ち進めた、空きマスが少なく合法手が複数ある終盤の局面
        let mut random = RandomAi::with_seed(3);
        let mut board = Board::new();
        while board.count(Color::Black) + board.count(Color::White) < 58 || board.legal_moves().len() < 2 {
            let mv = random.select_move(&board).expect("game continues");
            board = board.play(mv).unwrap();
            assert!(!board.is_game_over(), "pick another seed");
        }
        let evaluator = Arc::new(BatchRecorder::default());
        let config = MctsConfig { limit: SearchLimit::Iterations(500), batch_size: 8, ..Default::default() };
        let mut mcts = Mcts::with_evaluator(config, evaluator.clone());
        mcts.search(&board).unwrap();
        let stats = mcts.last_stats();
        assert_eq!(stats.iterations, 500);
        assert_eq!(stats.children.iter().map(|c| c.1).sum::<u32>(), 500);
        // 終局に届いた反復の分だけ、評価関数に渡された局面は少ない
        assert!(evaluator.evaluated.load(Ordering::Relaxed) < 500);
        assert!(stats.children.iter().all(|c| (0.0..=1.0).contains(&c.2)));
    }

    #[test]
    fn batched_time_limited_search_stops_in_time() {
        let config = MctsConfig {
            limit: SearchLimit::Time(Duration::from_millis(200)),
            threads: 4,
            batch_size: 16,
            ..Default::default()
        };
        let mut mcts = mcts_with(config, PositionalEvaluator::default(), 1);
        let start = Instant::now();
        mcts.search(&Board::new()).unwrap();
        assert!(start.elapsed() < Duration::from_millis(400), "{:?}", start.elapsed());
        assert!(mcts.last_stats().iterations > 100);
    }

    /// 葉をまとめて評価しても十分に強い（結果を再現できるよう 1 スレッドで行う）
    #[test]
    fn batched_search_beats_random_player() {
        let config = MctsConfig { limit: SearchLimit::Iterations(500), threads: 1, batch_size: 8, ..Default::default() };
        let mut wins = 0;
        for seed in 0..10 {
            let mut mcts = mcts_with(config, PositionalEvaluator::default(), seed);
            let mut random = RandomAi::with_seed(seed + 100);
            let (result, color) = if seed % 2 == 0 {
                (play_game(&mut mcts, &mut random), Color::Black)
            } else {
                (play_game(&mut random, &mut mcts), Color::White)
            };
            if result == GameResult::Win(color) {
                wins += 1;
            }
        }
        assert!(wins >= 8, "batched MCTS won only {wins}/10 games against random");
    }

    #[test]
    fn positional_evaluator_beats_random_player() {
        let config = MctsConfig { limit: SearchLimit::Iterations(500), ..Default::default() };
        let mut wins = 0;
        for seed in 0..10 {
            let mut mcts = mcts_with(config, PositionalEvaluator::default(), seed);
            let mut random = RandomAi::with_seed(seed + 100);
            let (result, color) = if seed % 2 == 0 {
                (play_game(&mut mcts, &mut random), Color::Black)
            } else {
                (play_game(&mut random, &mut mcts), Color::White)
            };
            if result == GameResult::Win(color) {
                wins += 1;
            }
        }
        assert!(wins >= 8, "MCTS(positional) won only {wins}/10 games against random");
    }

    #[test]
    fn beats_random_player() {
        for (parallel, threads) in [(ParallelMode::Tree, 1), (ParallelMode::Tree, 4), (ParallelMode::Root, 4)] {
            let config = MctsConfig { limit: SearchLimit::Iterations(500), threads, parallel, ..Default::default() };
            let mut wins = 0;
            for seed in 0..10 {
                let mut mcts = Mcts::with_seed(config, seed);
                let mut random = RandomAi::with_seed(seed + 100);
                let (result, color) = if seed % 2 == 0 {
                    (play_game(&mut mcts, &mut random), Color::Black)
                } else {
                    (play_game(&mut random, &mut mcts), Color::White)
                };
                if result == GameResult::Win(color) {
                    wins += 1;
                }
            }
            assert!(wins >= 8, "{parallel:?} x{threads}: MCTS won only {wins}/10 games against random");
        }
    }
}
