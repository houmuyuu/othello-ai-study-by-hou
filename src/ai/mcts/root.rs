//! ルート並列化
//!
//! 各スレッドが独立した探索木を持ち、探索後にルートの各手の訪問回数と評価値を合算する。
//! 各スレッドの木は探索後も [`Forest`] として保持し、次の探索で再利用できる。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;

use super::{
    Evaluator, MAX_REUSE_DEPTH, MAX_ROOT_VISITS, MctsConfig, SearchLimit, SearchOutput, TIME_CHECK_INTERVAL, evaluate,
    value_for,
};
use crate::board::{Board, Color, Move};
use crate::rng::Rng;

struct Node {
    board: Board,
    /// このノードへ至った手を指したプレイヤー（評価値はこの視点で蓄積する）
    mover: Color,
    mv: Option<Move>,
    parent: Option<usize>,
    children: Vec<usize>,
    untried: Vec<Move>,
    visits: u32,
    /// 評価値（0〜1）の合計
    value_sum: f64,
}

impl Node {
    fn new(board: Board, mv: Option<Move>, parent: Option<usize>) -> Self {
        Node {
            board,
            mover: board.turn().opponent(),
            mv,
            parent,
            children: Vec::new(),
            untried: board.legal_moves(),
            visits: 0,
            value_sum: 0.0,
        }
    }
}

/// スレッドごとの探索木の集まり（探索をまたいで保持する）
pub(super) struct Forest {
    root_board: Board,
    trees: Vec<Vec<Node>>,
}

impl Forest {
    pub(super) fn new(root_board: Board) -> Self {
        Forest { root_board, trees: Vec::new() }
    }

    pub(super) fn nodes(&self) -> usize {
        self.trees.iter().map(Vec::len).sum()
    }

    /// 各木について `board` に一致する子孫ノード（`MAX_REUSE_DEPTH` 手以内）を新しいルートにする。
    /// 見つからなかった木は破棄し、1 本も残らなければ `None`。
    pub(super) fn reroot(self, board: &Board) -> Option<Forest> {
        let trees: Vec<Vec<Node>> = self.trees.into_iter().filter_map(|t| reroot_tree(t, board)).collect();
        if trees.is_empty() {
            return None;
        }
        Some(Forest { root_board: *board, trees })
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
        let threads = config.threads.max(1);
        // スレッド数に合わせて木の本数を調整する（足りない分は新しい木）
        let root_board = self.root_board;
        self.trees.truncate(threads as usize);
        self.trees.resize_with(threads as usize, || vec![Node::new(root_board, None, None)]);

        let workers: Vec<(Worker, Vec<Node>)> = std::mem::take(&mut self.trees)
            .into_iter()
            .enumerate()
            .map(|(i, tree)| {
                let i = i as u32;
                let worker = Worker {
                    config: *config,
                    evaluator,
                    rng: Rng::new(rng.next_u64()),
                    // 反復回数指定の場合は全スレッド合計が指定値になるよう分配する
                    iterations: match config.limit {
                        SearchLimit::Iterations(n) => Some(n / threads + u32::from(i < n % threads)),
                        SearchLimit::Time(_) => None,
                    },
                    max_nodes: (config.max_nodes / threads as usize).max(1),
                };
                (worker, tree)
            })
            .collect();

        let results: Vec<(SearchOutput, Vec<Node>)> = if threads == 1 {
            workers.into_iter().map(|(mut w, mut tree)| (w.run(&mut tree, start, stop), tree)).collect()
        } else {
            thread::scope(|scope| {
                let handles: Vec<_> = workers
                    .into_iter()
                    .map(|(mut w, mut tree)| scope.spawn(move || (w.run(&mut tree, start, stop), tree)))
                    .collect();
                handles.into_iter().map(|h| h.join().expect("search thread panicked")).collect()
            })
        };

        let mut output = SearchOutput { iterations: 0, playouts: 0, nodes: 0, root_children: Vec::new() };
        for (result, tree) in results {
            output.iterations += result.iterations;
            output.playouts += result.playouts;
            output.nodes += result.nodes;
            output.root_children.extend(result.root_children);
            self.trees.push(tree);
        }
        output
    }
}

fn reroot_tree(tree: Vec<Node>, target: &Board) -> Option<Vec<Node>> {
    let idx = find_descendant(&tree, 0, target, MAX_REUSE_DEPTH)?;
    if idx == 0 {
        return Some(tree);
    }
    // 新しいルート以下の部分木だけを、インデックスを振り直して詰め直す
    let mut old: Vec<Option<Node>> = tree.into_iter().map(Some).collect();
    let mut new: Vec<Node> = Vec::new();
    let mut queue = VecDeque::from([(idx, None)]);
    while let Some((old_idx, parent)) = queue.pop_front() {
        let mut node = old[old_idx].take().expect("tree is acyclic");
        let children = std::mem::take(&mut node.children);
        let new_idx = new.len();
        node.parent = parent;
        if parent.is_none() {
            node.mv = None;
        }
        new.push(node);
        if let Some(p) = parent {
            new[p].children.push(new_idx);
        }
        queue.extend(children.into_iter().map(|c| (c, Some(new_idx))));
    }
    Some(new)
}

fn find_descendant(tree: &[Node], idx: usize, target: &Board, depth: u32) -> Option<usize> {
    if tree[idx].board == *target {
        return Some(idx);
    }
    if depth == 0 {
        return None;
    }
    tree[idx].children.iter().find_map(|&c| find_descendant(tree, c, target, depth - 1))
}

/// 1 本の探索木を担当する探索器（スレッドごとに 1 つ）
struct Worker<'a> {
    config: MctsConfig,
    evaluator: &'a dyn Evaluator,
    rng: Rng,
    /// このワーカーが担当する反復回数（時間制限の場合は `None`）
    iterations: Option<u32>,
    /// このワーカーの木のノード数上限
    max_nodes: usize,
}

impl Worker<'_> {
    fn run(&mut self, tree: &mut Vec<Node>, start: Instant, stop: Option<&AtomicBool>) -> SearchOutput {
        let mut iterations = 0u32;
        let mut playouts = 0u64;

        loop {
            let stopped = stop.is_some_and(|s| s.load(Ordering::Relaxed)) || tree[0].visits >= MAX_ROOT_VISITS;
            let done = stopped
                || match (self.iterations, self.config.limit) {
                (Some(n), _) => iterations >= n,
                (None, SearchLimit::Time(t)) => {
                    iterations.is_multiple_of(TIME_CHECK_INTERVAL) && start.elapsed() >= t
                }
                (None, SearchLimit::Iterations(_)) => unreachable!(),
            };
            if done {
                break;
            }

            let leaf = self.select_and_expand(tree);
            let (value_black, n) = evaluate(&tree[leaf].board, self.evaluator, &mut self.rng);
            playouts += n as u64;
            backpropagate(tree, leaf, value_black);
            iterations += 1;
        }

        let root_children = tree[0]
            .children
            .iter()
            .map(|&c| (tree[c].mv.unwrap(), tree[c].visits, tree[c].value_sum))
            .collect();
        SearchOutput { iterations, playouts, nodes: tree.len(), root_children }
    }

    /// 選択：UCB1 で子をたどり、未展開の手があればそれを 1 つ展開して返す。
    /// ノード数が上限に達していれば展開せずに既存の子をたどり、葉に着いたらそれを返す。
    fn select_and_expand(&mut self, tree: &mut Vec<Node>) -> usize {
        let mut idx = 0;
        loop {
            let full = tree.len() >= self.max_nodes && idx != 0;
            if !tree[idx].untried.is_empty() && !full {
                let k = self.rng.below(tree[idx].untried.len() as u32) as usize;
                let mv = tree[idx].untried.swap_remove(k);
                let board = tree[idx].board.play(mv).expect("legal move");
                let child = tree.len();
                tree.push(Node::new(board, Some(mv), Some(idx)));
                tree[idx].children.push(child);
                return child;
            }
            if tree[idx].children.is_empty() {
                // 終局ノード、またはノード数上限で展開できない葉
                return idx;
            }
            idx = self.best_uct_child(tree, idx);
        }
    }

    fn best_uct_child(&self, tree: &[Node], idx: usize) -> usize {
        let ln_n = (tree[idx].visits.max(1) as f64).ln();
        let c = self.config.exploration;
        *tree[idx]
            .children
            .iter()
            .max_by(|&&a, &&b| {
                let ucb = |i: usize| {
                    let n = tree[i].visits as f64;
                    tree[i].value_sum / n + c * (ln_n / n).sqrt()
                };
                ucb(a).total_cmp(&ucb(b))
            })
            .unwrap()
    }
}

fn backpropagate(tree: &mut [Node], leaf: usize, value_black: f64) {
    let mut idx = Some(leaf);
    while let Some(i) = idx {
        let node = &mut tree[i];
        node.visits += 1;
        node.value_sum += value_for(node.mover, value_black);
        idx = node.parent;
    }
}
