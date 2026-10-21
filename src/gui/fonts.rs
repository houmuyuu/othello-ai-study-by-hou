//! 日本語フォントの読み込み
//!
//! egui の組み込みフォントには日本語が含まれないため、OS のフォントを実行時に探して追加する。

use std::sync::Arc;

use eframe::egui::{self, FontData, FontDefinitions, FontFamily};

const CANDIDATES: &[&str] = &[
    // Windows
    "C:\\Windows\\Fonts\\YuGothM.ttc",
    "C:\\Windows\\Fonts\\meiryo.ttc",
    "C:\\Windows\\Fonts\\msgothic.ttc",
    // Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    // macOS
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
];

pub fn install_japanese_font(ctx: &egui::Context) {
    let Some(bytes) = CANDIDATES.iter().find_map(|path| std::fs::read(path).ok()) else {
        eprintln!("日本語フォントが見つかりませんでした。日本語が正しく表示されない可能性があります。");
        return;
    };
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert("japanese".to_owned(), Arc::new(FontData::from_owned(bytes)));
    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push("japanese".to_owned());
    }
    ctx.set_fonts(fonts);
}
