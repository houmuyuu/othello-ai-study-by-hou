//! 盤面の描画とクリック判定

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};

use crate::board::{BitIter, Board, Color, Move};

const BOARD_GREEN: Color32 = Color32::from_rgb(0x1f, 0x7a, 0x3a);
const LINE: Color32 = Color32::from_rgb(0x0e, 0x3d, 0x1c);
const HINT: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 60);
const LAST_MOVE: Color32 = Color32::from_rgb(0xe0, 0x40, 0x40);

/// 盤面を描画し、クリックされたマスを返す。
/// `show_hints` が真なら合法手を薄く表示する（人間の手番用）。
pub fn board_view(ui: &mut egui::Ui, board: &Board, last_move: Option<Move>, show_hints: bool) -> Option<u8> {
    let side = ui.available_width().min(ui.available_height()).max(200.0);
    let (response, painter) = ui.allocate_painter(Vec2::splat(side), Sense::click());
    let outer = response.rect;

    let margin = side * 0.06;
    let cell = (side - margin * 2.0) / 8.0;
    let origin = outer.min + Vec2::splat(margin);
    let grid = Rect::from_min_size(origin, Vec2::splat(cell * 8.0));
    let cell_center = |sq: u8| origin + Vec2::new((sq % 8) as f32 + 0.5, (sq / 8) as f32 + 0.5) * cell;

    painter.rect_filled(grid, 2.0, BOARD_GREEN);
    let line = Stroke::new(1.5, LINE);
    for i in 0..=8 {
        let d = i as f32 * cell;
        painter.line_segment([Pos2::new(grid.left() + d, grid.top()), Pos2::new(grid.left() + d, grid.bottom())], line);
        painter.line_segment([Pos2::new(grid.left(), grid.top() + d), Pos2::new(grid.right(), grid.top() + d)], line);
    }
    for (x, y) in [(2, 2), (6, 2), (2, 6), (6, 6)] {
        painter.circle_filled(origin + Vec2::new(x as f32, y as f32) * cell, cell * 0.06, LINE);
    }

    // 座標
    let coord_font = FontId::proportional(margin * 0.55);
    let coord_color = ui.visuals().text_color();
    for i in 0..8 {
        let c = (i as f32 + 0.5) * cell;
        let col = ((b'a' + i) as char).to_string();
        painter.text(Pos2::new(grid.left() + c, outer.top() + margin * 0.5), Align2::CENTER_CENTER, &col, coord_font.clone(), coord_color);
        painter.text(Pos2::new(outer.left() + margin * 0.5, grid.top() + c), Align2::CENTER_CENTER, (i + 1).to_string(), coord_font.clone(), coord_color);
    }

    // 石
    for sq in 0..64 {
        let Some(color) = board.get(sq) else { continue };
        let (fill, edge) = match color {
            Color::Black => (Color32::from_gray(20), Color32::from_gray(70)),
            Color::White => (Color32::from_gray(240), Color32::from_gray(150)),
        };
        let center = cell_center(sq);
        painter.circle_filled(center, cell * 0.42, fill);
        painter.circle_stroke(center, cell * 0.42, Stroke::new(1.0, edge));
    }

    if let Some(Move::Place(sq)) = last_move {
        painter.circle_filled(cell_center(sq), cell * 0.1, LAST_MOVE);
    }

    if show_hints {
        for sq in BitIter(board.legal_mask()) {
            painter.circle_filled(cell_center(sq), cell * 0.14, HINT);
        }
    }

    let pos = response.interact_pointer_pos().filter(|_| response.clicked())?;
    if !grid.contains(pos) {
        return None;
    }
    let rel = (pos - origin) / cell;
    let (col, row) = ((rel.x as u8).min(7), (rel.y as u8).min(7));
    Some(row * 8 + col)
}
