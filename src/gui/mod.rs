//! 対局画面（AI 同士・人と AI・人同士で共通）

mod board_view;
mod fonts;

use std::sync::Arc;
use std::time::Duration;

use eframe::egui;

use crate::ai::ParallelMode;
use crate::board::{Color, GameResult, Move};
use crate::game::{Game, Status, Waker};
use crate::nn::{inference_device, inference_device_if_ready};
use crate::player::{EvaluatorChoice, MctsParams, PlayerKind, auto_nn_batch_size, available_threads, resolve_model_path};

const COLORS: [Color; 2] = [Color::Black, Color::White];

fn color_name(color: Color) -> &'static str {
    match color {
        Color::Black => "黒",
        Color::White => "白",
    }
}

/// 1 人分の対局者設定。種類を切り替えても、種類ごとのパラメータは保持する。
struct PlayerSetup {
    choices: Vec<PlayerKind>,
    selected: usize,
}

impl PlayerSetup {
    fn new(initial: &str) -> Self {
        let choices = PlayerKind::choices();
        let selected = choices.iter().position(|c| c.label() == initial).unwrap_or(0);
        PlayerSetup { choices, selected }
    }

    fn current(&self) -> &PlayerKind {
        &self.choices[self.selected]
    }

    fn ui(&mut self, ui: &mut egui::Ui, id: usize) {
        let PlayerSetup { choices, selected } = self;
        egui::ComboBox::from_id_salt(("player_kind", id))
            .selected_text(choices[*selected].label())
            .show_ui(ui, |ui| {
                for (i, choice) in choices.iter().enumerate() {
                    ui.selectable_value(selected, i, choice.label());
                }
            });
        // 先手・後手で同じ UI を使うので、ウィジェットの状態が混ざらないよう ID を分ける
        ui.push_id(("player_params", id), |ui| params_ui(ui, &mut choices[*selected]));
    }
}

/// アルゴリズムごとのパラメータ編集 UI
fn params_ui(ui: &mut egui::Ui, kind: &mut PlayerKind) {
    match kind {
        PlayerKind::Human(params) => {
            ui.checkbox(&mut params.allow_ponder, "相手 AI の先読み（ponder）を許可")
                .on_hover_text("自分の手番中に、相手 AI が探索を進めることを許可します（相手が対応 AI の場合のみ）");
        }
        PlayerKind::Random => {
            ui.weak("パラメータなし");
        }
        PlayerKind::Mcts(params) => mcts_params_ui(ui, params),
    }
}

fn mcts_params_ui(ui: &mut egui::Ui, params: &mut MctsParams) {
    ui.add(
        egui::Slider::new(&mut params.think_time_secs, 0.1..=30.0)
            .logarithmic(true)
            .suffix(" 秒")
            .text("思考時間"),
    );
    ui.horizontal(|ui| {
        ui.label("評価関数");
        ui.radio_value(&mut params.evaluator, EvaluatorChoice::Playout, "プレイアウト")
            .on_hover_text("ランダムプレイアウトの結果の平均");
        ui.radio_value(&mut params.evaluator, EvaluatorChoice::Positional, "静的評価")
            .on_hover_text("マスの重みと着手可能数から勝率を推定（プレイアウトしない）");
        ui.radio_value(&mut params.evaluator, EvaluatorChoice::Nn, "NN")
            .on_hover_text("強化学習した価値ネットワーク（GPU が使えれば GPU、なければ CPU で推論）");
    });
    // 評価関数ごとのパラメータ
    match params.evaluator {
        EvaluatorChoice::Playout => {
            ui.add(
                egui::Slider::new(&mut params.playouts_per_eval, 1..=1000)
                    .logarithmic(true)
                    .suffix(" 回")
                    .text("プレイアウト"),
            );
        }
        EvaluatorChoice::Positional => {}
        EvaluatorChoice::Nn => {
            ui.horizontal(|ui| {
                ui.label("モデル");
                // 1 文字ごとに AI を作り直さないよう、入力確定（フォーカスが外れた時）に反映する
                let id = ui.id().with("model_path");
                let mut text = ui.data(|d| d.get_temp::<String>(id)).unwrap_or_else(|| params.model_path.clone());
                let response = ui.text_edit_singleline(&mut text);
                if response.lost_focus() {
                    params.model_path = text.clone();
                }
                if !response.has_focus() && !response.lost_focus() {
                    text = params.model_path.clone();
                }
                ui.data_mut(|d| d.insert_temp(id, text));
            });
            if !resolve_model_path(&params.model_path).exists() {
                ui.colored_label(ui.visuals().warn_fg_color, "モデルファイルが見つかりません");
            }
            ui.add_enabled_ui(params.parallel == ParallelMode::Tree, |ui| nn_batch_ui(ui, params));
            device_ui(ui);
        }
    }
    ui.add(
        egui::Slider::new(&mut params.threads, 1..=available_threads())
            .suffix(" 本")
            .text("スレッド数"),
    );
    ui.add_enabled_ui(params.threads > 1, |ui| {
        ui.horizontal(|ui| {
            ui.label("並列化");
            ui.radio_value(&mut params.parallel, ParallelMode::Tree, "木")
                .on_hover_text("全スレッドで 1 本の木を共有（バーチャルロス）");
            ui.radio_value(&mut params.parallel, ParallelMode::Root, "ルート")
                .on_hover_text("スレッドごとに独立した木で探索し、終了時に合算");
        });
    });
    ui.checkbox(&mut params.reuse_tree, "探索木を再利用")
        .on_hover_text("前の手の探索結果を引き継ぎます。無効にすると ponder もしません");
}

/// NN 評価時の、1 回の推論にまとめる葉の数の設定
fn nn_batch_ui(ui: &mut egui::Ui, params: &mut MctsParams) {
    let hover = "1 スレッドが 1 回の推論にまとめる葉の数。大きいほど推論はまとまって速くなるが、\
                 古い統計で葉を選ぶ分だけ探索の質は下がる（並列化が「木」のときだけ有効）";
    ui.horizontal(|ui| {
        let mut auto = params.nn_batch_size.is_none();
        if ui.checkbox(&mut auto, "バッチ自動").on_hover_text("GPU なら 32、CPU なら 1").changed() {
            // 手動に切り替えたら、その時点の自動の値から始める
            params.nn_batch_size = if auto {
                None
            } else {
                Some(inference_device_if_ready().map_or(1, |d| auto_nn_batch_size(d.is_gpu)))
            };
        }
        match &mut params.nn_batch_size {
            Some(n) => {
                ui.add(egui::Slider::new(n, 1..=128).logarithmic(true).suffix(" 局面")).on_hover_text(hover);
            }
            None => match inference_device_if_ready() {
                Some(d) => {
                    ui.label(format!("{} 局面", auto_nn_batch_size(d.is_gpu))).on_hover_text(hover);
                }
                None => {
                    ui.weak("デバイス確認中…");
                }
            },
        }
    });
}

/// NN の推論デバイスの表示
fn device_ui(ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.label("推論デバイス");
        match inference_device_if_ready() {
            None => {
                ui.spinner();
                ui.weak("確認中…（GPU の初回は準備に時間がかかります）");
            }
            Some(d) if d.is_gpu => {
                ui.strong(&d.label);
            }
            Some(d) => {
                ui.strong(&d.label);
                if let Some(reason) = &d.fallback_reason {
                    ui.weak(reason);
                }
            }
        }
    });
}

pub struct OthelloApp {
    game: Game,
    players: [PlayerSetup; 2],
    waker: Waker,
}

impl OthelloApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        fonts::install_japanese_font(&cc.egui_ctx);
        crate::panic_log::install();
        // GPU の確認（と準備）には時間がかかるので、画面を止めないよう起動時に裏で済ませておく
        let probe_ctx = cc.egui_ctx.clone();
        std::thread::spawn(move || {
            inference_device();
            probe_ctx.request_repaint();
        });
        let ctx = cc.egui_ctx.clone();
        OthelloApp {
            game: Game::new(),
            players: [PlayerSetup::new("MCTS"), PlayerSetup::new("ランダム")],
            waker: Arc::new(move || ctx.request_repaint()),
        }
    }

    fn player_kinds(&self) -> [&PlayerKind; 2] {
        [self.players[0].current(), self.players[1].current()]
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        for (i, label) in ["先手（黒）", "後手（白）"].into_iter().enumerate() {
            ui.heading(label);
            self.players[i].ui(ui, i);
            ui.separator();
        }

        ui.heading("対局");
        let status = self.game.status();
        ui.horizontal(|ui| {
            let start_label = if status == Status::Paused { "再開" } else { "開始" };
            if ui.add_enabled(matches!(status, Status::Ready | Status::Paused), egui::Button::new(start_label)).clicked() {
                self.game.start();
            }
            if ui.add_enabled(status == Status::Running, egui::Button::new("一時停止")).clicked() {
                self.game.pause();
            }
            if ui.button("リセット").clicked() {
                self.game.reset();
            }
        });
        let mut delay = self.game.move_delay.as_secs_f64();
        if ui.add(egui::Slider::new(&mut delay, 0.0..=3.0).suffix(" 秒").text("着手間隔")).changed() {
            self.game.move_delay = Duration::from_secs_f64(delay);
        }
        ui.weak("設定の変更は次の手から反映されます");
        ui.separator();
        ui.weak(format!("バージョン v{}", crate::VERSION));
    }

    fn info_ui(&self, ui: &mut egui::Ui) {
        let board = self.game.board();
        let status_text = match self.game.status() {
            Status::Ready => "開始前".to_string(),
            Status::Running => format!("対局中（{}番）", color_name(board.turn())),
            Status::Paused => "一時停止中".to_string(),
            Status::Finished => match board.result() {
                Some(GameResult::Win(c)) => format!("終局：{}の勝ち", color_name(c)),
                Some(GameResult::Draw) => "終局：引き分け".to_string(),
                None => "終局".to_string(),
            },
        };
        ui.heading(status_text);
        if let Some(error) = self.game.error() {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        ui.separator();

        let kinds = self.player_kinds();
        for (i, &color) in COLORS.iter().enumerate() {
            let marker = if board.turn() == color && self.game.status() != Status::Finished { "▶ " } else { "" };
            ui.strong(format!("{marker}{}（{}）: {} 石", color_name(color), kinds[i].label(), board.count(color)));

            if let Some((thinking_color, elapsed)) = self.game.thinking()
                && thinking_color == color
            {
                let text = format!("思考中 {:.1} 秒", elapsed.as_secs_f64());
                match kinds[i].think_time() {
                    Some(limit) => {
                        let progress = (elapsed.as_secs_f32() / limit.as_secs_f32()).min(1.0);
                        ui.add(egui::ProgressBar::new(progress).text(text));
                    }
                    None => {
                        ui.label(text);
                    }
                }
            }
            if let Some((pondering_color, elapsed)) = self.game.pondering()
                && pondering_color == color
            {
                ui.label(format!("先読み中（ponder） {:.1} 秒", elapsed.as_secs_f64()));
            }
            if let Some(summary) = self.game.summary(color) {
                ui.label(summary);
            }
            ui.add_space(6.0);
        }
        ui.separator();

        ui.strong("棋譜");
        egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
            for (n, record) in self.game.history().iter().enumerate() {
                let mv = match record.mv {
                    Move::Pass => "パス".to_string(),
                    mv => mv.to_string(),
                };
                ui.monospace(format!(
                    "{:>3}. {} {:<4} {:>6.2}s",
                    n + 1,
                    color_name(record.color),
                    mv,
                    record.elapsed.as_secs_f64()
                ));
            }
        });
    }
}

impl eframe::App for OthelloApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let kinds = [self.players[0].current().clone(), self.players[1].current().clone()];
        self.game.update([&kinds[0], &kinds[1]], &self.waker);

        egui::Panel::left("settings").resizable(false).default_size(280.0).show(ui, |ui| {
            ui.add_space(6.0);
            self.settings_ui(ui);
        });
        egui::Panel::right("info").resizable(true).default_size(280.0).show(ui, |ui| {
            ui.add_space(6.0);
            self.info_ui(ui);
        });
        egui::CentralPanel::default().show(ui, |ui| {
            let human_turn = self.game.awaiting_human([&kinds[0], &kinds[1]]);
            let clicked = ui
                .centered_and_justified(|ui| {
                    board_view::board_view(ui, self.game.board(), self.game.last_move(), human_turn)
                })
                .inner;
            if let Some(sq) = clicked {
                self.game.human_move(Move::Place(sq), [&kinds[0], &kinds[1]]);
            }
        });

        // 思考中の進捗表示と、着手間隔の待ち明けのために再描画を予約する
        if self.game.thinking().is_some() {
            ui.ctx().request_repaint_after(Duration::from_millis(50));
        } else if self.game.pondering().is_some() {
            ui.ctx().request_repaint_after(Duration::from_millis(200));
        } else if self.game.status() == Status::Running {
            ui.ctx().request_repaint_after(self.game.time_until_next_move());
        }
    }
}
