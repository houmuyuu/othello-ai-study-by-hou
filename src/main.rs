// リリースビルドの Windows ではコンソールウィンドウを出さない
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use othello::gui::OthelloApp;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title(format!("オセロ v{}", othello::VERSION))
            .with_inner_size([1100.0, 700.0])
            .with_min_inner_size([800.0, 500.0]),
        ..Default::default()
    };
    eframe::run_native("othello", options, Box::new(|cc| Ok(Box::new(OthelloApp::new(cc)))))
}
