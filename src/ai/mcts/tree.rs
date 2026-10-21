//! 木の並列化（バーチャルロス付き）
//!
//! 全スレッドで 1 本の木を共有する。
//! - ノードの統計（訪問回数・評価値合計・バーチャルロス）はアトミック変数で持ち、選択と逆伝播はロックしない。
//! - 子ノードは 2 回目の訪問時にまとめて生成する（`OnceLock` で 1 度だけ）。初回訪問ではプレイアウトのみ行う。
//! - 選択中のスレッドは通過したノードにバーチャルロス（評価 0 の仮の訪問）を加え、
//!   他のスレッドが同じ経路に集中しないようにする。逆伝播時に取り除く。
//!
//! - 設定 `batch_size` が 2 以上なら、各スレッドは選択を `batch_size` 回続けて葉を集め、まとめて評価してから
//!   逆伝播する（バーチャルロスにより、同じスレッドの続く選択も別の経路に向かう）。
//!
//! 盤面はノードに持たせず、ルートから手を再生して求める（ノードを小さく保つため）。
//!
//! 木は探索後も [`Tree`] として保持し、次の探索で該当する部分木をルートにして再利用できる。

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

use super::{
    Evaluator, MAX_REUSE_DEPTH, MAX_ROOT_VISITS, MctsConfig, SearchLimit, SearchOutput, TIME_CHECK_INTERVAL, black_value,
    result_value, value_for,
};
use crate::board::{Board, Color, Move};
use crate::rng::Rng;

/// アトミックに加算できる f64
struct AtomicF64(AtomicU64);

impl AtomicF64 {
    fn new(v: f64) -> Self {
        AtomicF64(AtomicU64::new(v.to_bits()))
    }

    fn load(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }

    fn add(&self, v: f64) {
        let _ = self.0.try_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
            Some((f64::from_bits(bits) + v).to_bits())
        });
    }
}

struct Node {
    /// このノードへ至った手（ルートでは未使用）
    mv: Move,
    visits: AtomicU32,
    /// 探索中（逆伝播待ち）のスレッド数
    virtual_loss: AtomicU32,
    /// このノードへ至った手を指したプレイヤー視点の評価値合計
    value_sum: AtomicF64,
    children: OnceLock<Box<[Node]>>,
}

impl Node {
    fn new(mv: Move) -> Self {
        Node {
            mv,
            visits: AtomicU32::new(0),
            virtual_loss: AtomicU32::new(0),
            value_sum: AtomicF64::new(0.0),
            children: OnceLock::new(),
        }
    }
}

/// 探索木（探索をまたいで保持する）
pub(super) struct Tree {
    root_board: Board,
    root: Node,
    /// ノード数
    nodes: usize,
}

impl Tree {
    pub(super) fn new(root_board: Board) -> Self {
        Tree { root_board, root: Node::new(Move::Pass), nodes: 1 }
    }

    pub(super) fn nodes(&self) -> usize {
        self.nodes
    }

    /// `board` に一致する子孫ノード（`MAX_REUSE_DEPTH` 手以内）を新しいルートにした木を返す。
    /// 見つからなければ `None`。それ以外の部分は破棄される。
    pub(super) fn reroot(self, board: &Board) -> Option<Tree> {
        let (root_board, root) = find_descendant(self.root_board, self.root, board, MAX_REUSE_DEPTH)?;
        let nodes = count_nodes(&root);
        Some(Tree { root_board, root, nodes })
    }

    /// 探索する。`stop` が立った場合も終了する。
    pub(super) fn search(
        &mut self,
        config: &MctsConfig,
        evaluator: &dyn Evaluator,
        rng: &mut Rng,
        start: Instant,
        stop: Option<&AtomicBool>,
    ) -> SearchOutput {
        // ルートは必ず展開しておく（ノード数上限より優先）
        if self.root.children.get().is_none() {
            let moves = self.root_board.legal_moves();
            self.nodes += moves.len();
            let _ = self.root.children.set(moves.into_iter().map(Node::new).collect());
        }

        let shared = Shared {
            config,
            evaluator,
            root_board: self.root_board,
            root: &self.root,
            start,
            stop,
            nodes: AtomicUsize::new(self.nodes),
            claimed: AtomicU32::new(0),
        };

        let threads = config.threads.max(1);
        let seeds: Vec<u64> = (0..threads).map(|_| rng.next_u64()).collect();
        let results: Vec<(u32, u64)> = if threads == 1 {
            vec![run_worker(&shared, Rng::new(seeds[0]))]
        } else {
            thread::scope(|scope| {
                let handles: Vec<_> = seeds
                    .iter()
                    .map(|&seed| {
                        let shared = &shared;
                        scope.spawn(move || run_worker(shared, Rng::new(seed)))
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("search thread panicked")).collect()
            })
        };
        self.nodes = shared.nodes.load(Ordering::Relaxed);

        let root_children = self
            .root
            .children
            .get()
            .expect("root is expanded")
            .iter()
            .map(|c| (c.mv, c.visits.load(Ordering::Relaxed), c.value_sum.load()))
            .collect();
        SearchOutput {
            iterations: results.iter().map(|r| r.0).sum(),
            playouts: results.iter().map(|r| r.1).sum(),
            nodes: self.nodes,
            root_children,
        }
    }
}

fn find_descendant(board: Board, node: Node, target: &Board, depth: u32) -> Option<(Board, Node)> {
    if board == *target {
        return Some((board, node));
    }
    if depth == 0 {
        return None;
    }
    let children = node.children.into_inner()?;
    children.into_vec().into_iter().find_map(|child| {
        let next = board.play(child.mv)?;
        find_descendant(next, child, target, depth - 1)
    })
}

fn count_nodes(node: &Node) -> usize {
    1 + node.children.get().map_or(0, |c| c.iter().map(count_nodes).sum())
}

/// 全スレッドで共有する探索状態
struct Shared<'a> {
    config: &'a MctsConfig,
    evaluator: &'a dyn Evaluator,
    root_board: Board,
    root: &'a Node,
    start: Instant,
    /// 外部からの停止要求（ponder 用）
    stop: Option<&'a AtomicBool>,
    /// 生成済みノード数
    nodes: AtomicUsize,
    /// 反復回数指定時に、各スレッドが反復を 1 回ずつ確保するためのカウンタ
    claimed: AtomicU32,
}

/// 1 スレッド分の探索ループ。(反復回数, プレイアウト回数) を返す。
fn run_worker(shared: &Shared, mut rng: Rng) -> (u32, u64) {
    let batch_size = shared.config.batch_size.max(1);
    let mut iterations = 0u32;
    let mut playouts = 0u64;
    let mut batch = LeafBatch::default();
    loop {
        let stopped = shared.stop.is_some_and(|s| s.load(Ordering::Relaxed))
            || shared.root.visits.load(Ordering::Relaxed) >= MAX_ROOT_VISITS;
        if stopped {
            break;
        }
        // 今回の反復回数（反復回数指定なら、全スレッド合計が指定値を超えないよう確保する）
        let k = match shared.config.limit {
            SearchLimit::Iterations(n) => {
                let claimed = shared.claimed.fetch_add(batch_size, Ordering::Relaxed);
                if claimed >= n {
                    break;
                }
                batch_size.min(n - claimed)
            }
            SearchLimit::Time(t) => {
                // まとめて評価する場合は 1 回の反復が長いので、毎回確認する
                let check = batch_size > 1 || iterations.is_multiple_of(TIME_CHECK_INTERVAL);
                if check && shared.start.elapsed() >= t {
                    break;
                }
                batch_size
            }
        };
        playouts += iterate_batch(shared, &mut rng, k as usize, &mut batch) as u64;
        iterations += k;
    }
    (iterations, playouts)
}

/// 1 スレッドが使い回す、評価待ちの葉のバッファ
#[derive(Default)]
struct LeafBatch<'a> {
    /// 各反復で選んだ経路（ルートの子から葉まで）
    paths: Vec<Vec<(&'a Node, Color)>>,
    /// 評価関数で評価する葉の盤面
    boards: Vec<Board>,
    /// `boards` のそれぞれが、どの経路の葉か
    pending: Vec<usize>,
}

/// 選択→展開→評価→逆伝播 を `k` 回まとめて行い、実行したプレイアウト回数を返す。
///
/// 選択を `k` 回続けてから、終局していない葉をまとめて評価する。選択した経路のバーチャルロスは
/// 逆伝播まで残るので、続く選択は別の経路に向かい、葉が重ならない。
fn iterate_batch<'a>(shared: &'a Shared, rng: &mut Rng, k: usize, batch: &mut LeafBatch<'a>) -> u32 {
    if batch.paths.len() < k {
        batch.paths.resize_with(k, || Vec::with_capacity(64));
    }
    batch.boards.clear();
    batch.pending.clear();

    for i in 0..k {
        let path = &mut batch.paths[i];
        let board = select_leaf(shared, rng, path);
        match board.result() {
            // 終局局面は評価関数を使わず、すぐに逆伝播する
            Some(result) => backpropagate(shared, path, result_value(result, Color::Black)),
            None => {
                batch.boards.push(board);
                batch.pending.push(i);
            }
        }
    }
    if batch.boards.is_empty() {
        return 0;
    }

    let values = shared.evaluator.evaluate_batch(&batch.boards, rng);
    assert_eq!(values.len(), batch.boards.len(), "evaluate_batch must return one value per board");
    for ((&i, board), value) in batch.pending.iter().zip(&batch.boards).zip(values) {
        backpropagate(shared, &batch.paths[i], black_value(board, value));
    }
    shared.evaluator.playouts_per_eval() * batch.boards.len() as u32
}

/// ルートから評価する葉まで選択し、葉の盤面を返す。通過した経路を `path` に入れ、
/// 各ノードにバーチャルロスを加える（[`backpropagate`] で取り除く）。
fn select_leaf<'a>(shared: &'a Shared, rng: &mut Rng, path: &mut Vec<(&'a Node, Color)>) -> Board {
    path.clear();
    let mut node = shared.root;
    let mut board = shared.root_board;
    // このスレッドが、まだ誰も評価していないノードに最初に到達したか
    let mut first_arrival = false;

    loop {
        let children = match node.children.get() {
            Some(children) => children,
            None => {
                if first_arrival || board.is_game_over() {
                    return board;
                }
                match expand(shared, node, &board) {
                    Some(children) => children,
                    // ノード数の上限に達したので、この葉を評価する
                    None => return board,
                }
            }
        };
        let child = select_child(children, shared.config.exploration, rng);
        let prev_virtual_loss = child.virtual_loss.fetch_add(1, Ordering::Relaxed);
        first_arrival = prev_virtual_loss == 0 && child.visits.load(Ordering::Relaxed) == 0;
        path.push((child, board.turn()));
        board = board.play(child.mv).expect("legal move");
        node = child;
    }
}

/// 黒視点の評価値 `value_black` を経路に加え、バーチャルロスを取り除く
fn backpropagate(shared: &Shared, path: &[(&Node, Color)], value_black: f64) {
    for &(node, mover) in path {
        node.value_sum.add(value_for(mover, value_black));
        node.visits.fetch_add(1, Ordering::Relaxed);
        node.virtual_loss.fetch_sub(1, Ordering::Relaxed);
    }
    shared.root.visits.fetch_add(1, Ordering::Relaxed);
}

/// 子ノードをまとめて生成する。ノード数の上限を超える場合は生成せず、
/// 他のスレッドが既に展開していればそれを返す。
fn expand<'a>(shared: &Shared, node: &'a Node, board: &Board) -> Option<&'a [Node]> {
    let moves = board.legal_moves();
    let n = moves.len();
    if shared.nodes.fetch_add(n, Ordering::Relaxed) + n > shared.config.max_nodes {
        shared.nodes.fetch_sub(n, Ordering::Relaxed);
        return node.children.get().map(|c| &**c);
    }
    let mut created = false;
    let children = node.children.get_or_init(|| {
        created = true;
        moves.into_iter().map(Node::new).collect()
    });
    if !created {
        // 他のスレッドが先に展開したので、確保した分を返却する
        shared.nodes.fetch_sub(n, Ordering::Relaxed);
    }
    Some(children)
}

/// バーチャルロスを含めた UCB1 で子を選ぶ。未訪問の子は最優先（複数あればランダム）。
fn select_child<'a>(children: &'a [Node], exploration: f64, rng: &mut Rng) -> &'a Node {
    let count = |c: &Node| c.visits.load(Ordering::Relaxed) + c.virtual_loss.load(Ordering::Relaxed);
    let total: u32 = children.iter().map(count).sum();
    let ln_n = (total.max(1) as f64).ln();

    let len = children.len();
    let offset = rng.below(len as u32) as usize;
    let mut best = offset;
    let mut best_score = f64::NEG_INFINITY;
    for i in 0..len {
        let k = (offset + i) % len;
        let n = count(&children[k]);
        let score = if n == 0 {
            f64::INFINITY
        } else {
            // バーチャルロス分は評価 0 の訪問として扱われる
            children[k].value_sum.load() / n as f64 + exploration * (ln_n / n as f64).sqrt()
        };
        if score > best_score {
            best = k;
            best_score = score;
        }
    }
    &children[best]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::mcts::PositionalEvaluator;

    fn assert_no_virtual_loss(node: &Node) {
        assert_eq!(node.virtual_loss.load(Ordering::Relaxed), 0);
        if let Some(children) = node.children.get() {
            children.iter().for_each(assert_no_virtual_loss);
        }
    }

    /// 葉をまとめて評価しても、探索後にはバーチャルロスがすべて取り除かれている
    #[test]
    fn virtual_loss_is_removed_after_batched_search() {
        for (threads, batch_size) in [(1, 1), (1, 16), (4, 16)] {
            let config = MctsConfig { limit: SearchLimit::Iterations(2000), threads, batch_size, ..Default::default() };
            let mut tree = Tree::new(Board::new());
            let output = tree.search(&config, &PositionalEvaluator::default(), &mut Rng::new(1), Instant::now(), None);
            assert_eq!(output.iterations, 2000);
            assert_eq!(tree.root.visits.load(Ordering::Relaxed), 2000);
            assert_no_virtual_loss(&tree.root);
        }
    }
}
