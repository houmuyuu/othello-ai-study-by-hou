//! 対局の進行管理（GUI に依存しない）
//!
//! AI の思考は別スレッドで行い、`update` を毎フレーム呼ぶことで結果を取り込む。
//! 人間の手番では `human_move` で着手を受け付ける。
//!
//! AI は対局中保持し続ける（探索木を次の手に再利用するため）。設定が変わったときだけ作り直す。
//! 人間が許可している場合（`allow_ponder`）、人間の手番中に相手 AI が先読み（ponder）する。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use crate::ai::Ai;
use crate::board::{Board, Color, Move};
use crate::player::PlayerKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// 開始前
    Ready,
    Running,
    Paused,
    Finished,
}

#[derive(Debug, Clone)]
pub struct MoveRecord {
    pub color: Color,
    pub mv: Move,
    pub elapsed: Duration,
}

type BoxedAi = Box<dyn Ai + Send>;

struct ThinkResult {
    ai: BoxedAi,
    mv: Option<Move>,
    summary: Option<String>,
}

struct Thinking {
    color: Color,
    started: Instant,
    rx: Receiver<ThinkResult>,
}

struct Pondering {
    /// ponder している AI の色
    color: Color,
    started: Instant,
    stop: Arc<AtomicBool>,
    rx: Receiver<BoxedAi>,
}

impl Drop for Pondering {
    /// リセット等で破棄されたときに、ponder スレッドが走り続けないよう止める
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// 対局者 1 人分の AI。思考・ponder 中はスレッドに貸し出しているため `ai` は `None`。
struct AiSlot {
    /// この AI を作ったときの設定
    kind: PlayerKind,
    ai: Option<BoxedAi>,
}

/// AI の思考完了時に呼ばれるコールバック（GUI の再描画要求など）
pub type Waker = Arc<dyn Fn() + Send + Sync>;

pub struct Game {
    board: Board,
    history: Vec<MoveRecord>,
    status: Status,
    ais: [Option<AiSlot>; 2],
    thinking: Option<Thinking>,
    pondering: Option<Pondering>,
    /// 人間の手番になった時刻（消費時間の計測用）
    turn_started: Instant,
    last_move_at: Option<Instant>,
    /// 各色の直前の思考内容
    summaries: [Option<String>; 2],
    /// AI を作れなかった場合などのエラー（対局は一時停止する）
    error: Option<String>,
    /// 着手と着手の間の最小間隔（AI 同士の対戦を目で追えるようにする）
    pub move_delay: Duration,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

fn color_index(color: Color) -> usize {
    match color {
        Color::Black => 0,
        Color::White => 1,
    }
}

impl Game {
    pub fn new() -> Self {
        Game {
            board: Board::new(),
            history: Vec::new(),
            status: Status::Ready,
            ais: [None, None],
            thinking: None,
            pondering: None,
            turn_started: Instant::now(),
            last_move_at: None,
            summaries: [None, None],
            error: None,
            move_delay: Duration::from_millis(300),
        }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn history(&self) -> &[MoveRecord] {
        &self.history
    }

    pub fn status(&self) -> Status {
        self.status
    }

    pub fn last_move(&self) -> Option<Move> {
        self.history.last().map(|r| r.mv)
    }

    pub fn summary(&self, color: Color) -> Option<&str> {
        self.summaries[color_index(color)].as_deref()
    }

    /// 直近のエラー（AI を作れなかった等）
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// 思考中の色と経過時間
    pub fn thinking(&self) -> Option<(Color, Duration)> {
        self.thinking.as_ref().map(|t| (t.color, t.started.elapsed()))
    }

    /// ponder 中の AI の色と経過時間
    pub fn pondering(&self) -> Option<(Color, Duration)> {
        self.pondering.as_ref().map(|p| (p.color, p.started.elapsed()))
    }

    /// 開始／再開
    pub fn start(&mut self) {
        if matches!(self.status, Status::Ready | Status::Paused) {
            self.error = None;
            self.status = Status::Running;
            self.turn_started = Instant::now();
        }
    }

    /// 一時停止。思考中の AI はその手を指し終えてから止まる。ponder は止める。
    pub fn pause(&mut self) {
        if self.status == Status::Running {
            self.status = Status::Paused;
        }
    }

    /// 初期局面に戻す。思考中のスレッドの結果は破棄され、ponder は止まる。AI も作り直す。
    pub fn reset(&mut self) {
        let move_delay = self.move_delay;
        *self = Game::new();
        self.move_delay = move_delay;
    }

    /// 人間の手番で、着手を受け付けられる状態か
    pub fn awaiting_human(&self, players: [&PlayerKind; 2]) -> bool {
        self.status == Status::Running
            && self.thinking.is_none()
            && players[color_index(self.board.turn())].is_human()
    }

    /// 人間の着手。受け付けたら `true`。
    pub fn human_move(&mut self, mv: Move, players: [&PlayerKind; 2]) -> bool {
        if !self.awaiting_human(players) || self.board.play(mv).is_none() {
            return false;
        }
        let elapsed = self.turn_started.elapsed();
        self.apply(mv, elapsed);
        true
    }

    /// 進行処理。毎フレーム呼ぶ。`players` は [黒, 白]。
    pub fn update(&mut self, players: [&PlayerKind; 2], waker: &Waker) {
        if let Some(thinking) = &self.thinking {
            match thinking.rx.try_recv() {
                Ok(result) => {
                    let (color, elapsed) = (thinking.color, thinking.started.elapsed());
                    self.thinking = None;
                    self.return_ai(color, result.ai);
                    self.summaries[color_index(color)] = result.summary;
                    match result.mv {
                        Some(mv) => self.apply(mv, elapsed),
                        None => self.status = Status::Finished,
                    }
                }
                Err(TryRecvError::Empty) => return,
                // 思考スレッドが異常終了した場合は一時停止しておく（AI は次の手で作り直される）
                Err(TryRecvError::Disconnected) => {
                    let color = thinking.color;
                    self.thinking = None;
                    self.ais[color_index(color)] = None;
                    self.status = Status::Paused;
                    return;
                }
            }
        }

        if !self.update_pondering(players) {
            return;
        }

        if self.status != Status::Running {
            return;
        }
        if self.board.is_game_over() {
            self.status = Status::Finished;
            return;
        }

        let color = self.board.turn();
        let player = players[color_index(color)];
        if player.is_human() {
            // 人間に合法手がなければ自動でパスする
            if self.board.legal_moves() == [Move::Pass] {
                self.apply(Move::Pass, Duration::ZERO);
            } else {
                self.start_pondering(players, waker);
            }
            return;
        }

        if self.time_until_next_move() > Duration::ZERO {
            return;
        }
        let Some(mut ai) = self.take_ai(color, player) else { return };
        let (tx, rx) = mpsc::channel();
        let board = self.board;
        let waker = waker.clone();
        thread::spawn(move || {
            let mv = ai.select_move(&board);
            let summary = ai.summary();
            // 受信側がリセットで破棄されていれば送信は失敗するが問題ない
            let _ = tx.send(ThinkResult { ai, mv, summary });
            waker();
        });
        self.thinking = Some(Thinking { color, started: Instant::now(), rx });
    }

    /// ponder の継続判定と、終わった ponder からの AI の回収。
    /// AI の返却待ちで進行できなければ `false`。
    fn update_pondering(&mut self, players: [&PlayerKind; 2]) -> bool {
        let Some(pondering) = &self.pondering else { return true };
        let color = pondering.color;
        let human = players[color_index(color.opponent())];
        let keep = self.status == Status::Running
            && self.board.turn() == color.opponent()
            && human.allows_opponent_ponder()
            && self.ais[color_index(color)].as_ref().is_some_and(|s| s.kind == *players[color_index(color)]);
        if !keep {
            pondering.stop.store(true, Ordering::Relaxed);
        }
        match pondering.rx.try_recv() {
            Ok(ai) => {
                self.pondering = None;
                self.return_ai(color, ai);
                true
            }
            Err(TryRecvError::Empty) => keep,
            Err(TryRecvError::Disconnected) => {
                self.pondering = None;
                self.ais[color_index(color)] = None;
                true
            }
        }
    }

    /// 人間の手番中、許可されていれば相手 AI の ponder を始める
    fn start_pondering(&mut self, players: [&PlayerKind; 2], waker: &Waker) {
        let human_color = self.board.turn();
        let ai_color = human_color.opponent();
        let ai_kind = players[color_index(ai_color)];
        if self.pondering.is_some() || !players[color_index(human_color)].allows_opponent_ponder() || !ai_kind.can_ponder()
        {
            return;
        }
        let Some(mut ai) = self.take_ai(ai_color, ai_kind) else { return };
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let board = self.board;
        let (thread_stop, waker) = (stop.clone(), waker.clone());
        thread::spawn(move || {
            ai.ponder(&board, &thread_stop);
            let _ = tx.send(ai);
            waker();
        });
        self.pondering = Some(Pondering { color: ai_color, started: Instant::now(), stop, rx });
    }

    /// 対局者の AI を借り出す。設定が変わっていれば作り直す。
    /// 他のスレッドに貸し出し中なら `None`。AI を作れなければエラーを記録して一時停止し、`None`。
    fn take_ai(&mut self, color: Color, kind: &PlayerKind) -> Option<BoxedAi> {
        let slot = &mut self.ais[color_index(color)];
        match slot {
            Some(s) if s.ai.is_none() => None,
            Some(s) if s.kind == *kind => s.ai.take(),
            _ => match kind.build_ai() {
                Ok(ai) => {
                    let ai = ai?;
                    *slot = Some(AiSlot { kind: kind.clone(), ai: None });
                    Some(ai)
                }
                Err(e) => {
                    *slot = None;
                    self.error = Some(e);
                    self.status = Status::Paused;
                    None
                }
            },
        }
    }

    fn return_ai(&mut self, color: Color, ai: BoxedAi) {
        if let Some(slot) = &mut self.ais[color_index(color)] {
            slot.ai = Some(ai);
        }
    }

    /// 次の着手を開始できるまでの残り時間（`move_delay` による待ち）
    pub fn time_until_next_move(&self) -> Duration {
        match self.last_move_at {
            Some(t) => self.move_delay.saturating_sub(t.elapsed()),
            None => Duration::ZERO,
        }
    }

    fn apply(&mut self, mv: Move, elapsed: Duration) {
        let color = self.board.turn();
        self.board = self.board.play(mv).expect("illegal move");
        self.history.push(MoveRecord { color, mv, elapsed });
        self.last_move_at = Some(Instant::now());
        self.turn_started = Instant::now();
        if self.board.is_game_over() {
            self.status = Status::Finished;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::player::{HumanParams, MctsParams};

    fn noop_waker() -> Waker {
        Arc::new(|| {})
    }

    fn human(allow_ponder: bool) -> PlayerKind {
        PlayerKind::Human(HumanParams { allow_ponder })
    }

    fn fast_mcts() -> PlayerKind {
        PlayerKind::Mcts(MctsParams { think_time_secs: 0.01, threads: 2, ..Default::default() })
    }

    /// 条件を満たすまで update を回す
    fn update_until(game: &mut Game, players: [&PlayerKind; 2], cond: impl Fn(&Game) -> bool) {
        let waker = noop_waker();
        let deadline = Instant::now() + Duration::from_secs(30);
        while !cond(game) {
            assert!(Instant::now() < deadline, "condition was not met");
            game.update(players, &waker);
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn ai_vs_ai_plays_to_the_end_without_pondering() {
        let mcts = fast_mcts();
        let mut game = Game::new();
        game.move_delay = Duration::ZERO;
        game.start();
        update_until(&mut game, [&mcts, &PlayerKind::Random], |g| {
            assert!(g.pondering().is_none(), "AI 同士では ponder しない");
            g.status() == Status::Finished
        });
        assert!(game.board().is_game_over());
        assert!(game.history().len() >= 9);
        assert!(game.summary(Color::Black).is_some());
    }

    fn ai_addr(ai: &BoxedAi) -> usize {
        &**ai as *const (dyn Ai + Send) as *const () as usize
    }

    /// 対局中は同じ AI を使い続け（探索木を引き継ぐため）、設定が変わったときだけ作り直す。
    /// 木の再利用そのものは MCTS 側のテストで確認している。
    #[test]
    fn ai_is_kept_until_settings_change() {
        let mcts = fast_mcts();
        let mut game = Game::new();

        let ai = game.take_ai(Color::Black, &mcts).unwrap();
        let addr = ai_addr(&ai);
        // 貸し出し中は借りられない
        assert!(game.take_ai(Color::Black, &mcts).is_none());
        game.return_ai(Color::Black, ai);
        // 同じ設定なら同じ AI（返却後も解放されていないのでアドレスで同一性を確認できる）
        let ai = game.take_ai(Color::Black, &mcts).unwrap();
        assert_eq!(ai_addr(&ai), addr);
        game.return_ai(Color::Black, ai);

        // 設定が変わると作り直される
        let changed = PlayerKind::Mcts(MctsParams { think_time_secs: 0.02, threads: 2, ..Default::default() });
        let ai = game.take_ai(Color::Black, &changed).unwrap();
        assert!(game.ais[0].as_ref().is_some_and(|s| s.kind == changed));
        game.return_ai(Color::Black, ai);

        // 人間には AI を作らない
        assert!(game.take_ai(Color::White, &human(true)).is_none());
    }

    #[test]
    fn human_moves_are_accepted_only_on_their_turn() {
        let (black, white) = (human(true), human(true));
        let players = [&black, &white];
        let mut game = Game::new();
        // 開始前は受け付けない
        assert!(!game.human_move(Move::Place(19), players));
        game.start();
        game.update(players, &noop_waker());
        assert!(!game.human_move(Move::Place(0), players)); // 非合法手
        assert!(game.human_move(Move::Place(19), players)); // d3
        assert_eq!(game.board().turn(), Color::White);
        assert!(game.pondering().is_none(), "人間同士では ponder しない");
    }

    #[test]
    fn ai_ponders_during_human_turn_only_when_allowed() {
        let mcts = fast_mcts();
        for allow in [true, false] {
            let black = human(allow);
            let players = [&black, &mcts];
            let mut game = Game::new();
            game.start();
            game.update(players, &noop_waker());
            assert_eq!(game.pondering().map(|p| p.0), allow.then_some(Color::White), "allow_ponder = {allow}");

            // 人間が考えている間に ponder が進む
            thread::sleep(Duration::from_millis(50));
            // 人間が指すと ponder は止まり、AI が思考して指す
            assert!(game.human_move(Move::Place(19), players));
            update_until(&mut game, players, |g| g.board().turn() == Color::Black && g.thinking().is_none());
            let summary = game.summary(Color::White).unwrap();
            // ponder した場合だけ、その木が再利用される
            assert_eq!(!summary.contains("うち再利用 0)"), allow, "{summary}");
        }
    }

    #[test]
    fn pondering_stops_when_permission_is_revoked() {
        let mcts = fast_mcts();
        let (allowed, denied) = (human(true), human(false));
        let mut game = Game::new();
        game.start();
        game.update([&allowed, &mcts], &noop_waker());
        assert!(game.pondering().is_some());
        update_until(&mut game, [&denied, &mcts], |g| g.pondering().is_none());
    }

    #[test]
    fn missing_model_pauses_game_with_error() {
        use crate::player::EvaluatorChoice;
        let nn = PlayerKind::Mcts(MctsParams {
            evaluator: EvaluatorChoice::Nn,
            model_path: "/nonexistent/model.bpk".to_string(),
            ..Default::default()
        });
        let mut game = Game::new();
        game.start();
        game.update([&nn, &PlayerKind::Random], &noop_waker());
        assert_eq!(game.status(), Status::Paused);
        assert!(game.error().is_some_and(|e| e.contains("/nonexistent/model.bpk")));
        assert!(game.thinking().is_none());
        // 再開するとエラー表示は消える
        game.start();
        assert!(game.error().is_none());
    }

    #[test]
    fn reset_discards_pending_thinking_and_pondering() {
        let slow = PlayerKind::Mcts(MctsParams { think_time_secs: 0.2, threads: 1, ..Default::default() });
        let mut game = Game::new();
        game.start();
        game.update([&slow, &slow], &noop_waker());
        assert!(game.thinking().is_some());
        game.reset();
        assert!(game.thinking().is_none());
        assert_eq!(game.status(), Status::Ready);
        assert!(game.history().is_empty());

        let black = human(true);
        game.start();
        game.update([&black, &slow], &noop_waker());
        let stop = game.pondering.as_ref().unwrap().stop.clone();
        game.reset();
        assert!(stop.load(Ordering::Relaxed), "リセットで ponder スレッドに停止を指示する");
    }
}
