//! 盤面 → ニューラルネットの入力特徴量
//!
//! 入力は 3 枚の 8×8 平面（手番側の石、相手の石、手番側の合法手）。
//! 手番側から見た表現にすることで、黒番・白番を同じネットワークで扱う。

use crate::board::{BitIter, Board};

/// 入力平面の数
pub const PLANES: usize = 3;
/// 1 局面あたりの特徴量の長さ（PLANES × 64）
pub const FEATURE_LEN: usize = PLANES * 64;

/// 盤面の特徴量を `out`（長さ `FEATURE_LEN`）に書き込む
pub fn write_features(board: &Board, out: &mut [f32]) {
    assert_eq!(out.len(), FEATURE_LEN);
    out.fill(0.0);
    let me = board.turn();
    let planes = [board.stones(me), board.stones(me.opponent()), board.legal_mask()];
    for (p, bits) in planes.into_iter().enumerate() {
        for sq in BitIter(bits) {
            out[p * 64 + sq as usize] = 1.0;
        }
    }
}

pub fn features(board: &Board) -> Vec<f32> {
    let mut out = vec![0.0; FEATURE_LEN];
    write_features(board, &mut out);
    out
}

/// 盤面の対称変換の数（回転 4 通り × 反転の有無）
pub const SYMMETRIES: usize = 8;

/// マス `sq` を対称変換 `sym`（0〜7）で移す
pub fn transform_square(sq: u8, sym: usize) -> u8 {
    let (r, c) = (sq / 8, sq % 8);
    let (r, c) = match sym {
        0 => (r, c),
        1 => (c, 7 - r),     // 90° 回転
        2 => (7 - r, 7 - c), // 180° 回転
        3 => (7 - c, r),     // 270° 回転
        4 => (r, 7 - c),     // 左右反転
        5 => (7 - r, c),     // 上下反転
        6 => (c, r),         // 主対角線で反転
        7 => (7 - c, 7 - r), // 反対角線で反転
        _ => panic!("symmetry index out of range: {sym}"),
    };
    r * 8 + c
}

fn transform_bits(bits: u64, sym: usize) -> u64 {
    BitIter(bits).fold(0, |acc, sq| acc | (1u64 << transform_square(sq, sym)))
}

/// 盤面を対称変換する（オセロのルールは対称変換で不変なので、評価値も変わらない）
pub fn transform_board(board: &Board, sym: usize) -> Board {
    use crate::board::Color;
    Board::from_bitboards(
        transform_bits(board.stones(Color::Black), sym),
        transform_bits(board.stones(Color::White), sym),
        board.turn(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Color, Move};

    #[test]
    fn features_are_from_side_to_move() {
        let board = Board::new();
        let f = features(&board);
        // 黒番：平面 0 が黒、平面 1 が白、平面 2 が合法手 4 つ
        assert_eq!(f[28], 1.0); // e4 黒
        assert_eq!(f[64 + 27], 1.0); // d4 白
        assert_eq!(f[128..].iter().sum::<f32>(), 4.0);

        let white_turn = board.play(Move::Place(19)).unwrap();
        let f = features(&white_turn);
        // 白番：平面 0 が白
        assert_eq!(f[..64].iter().sum::<f32>(), white_turn.count(Color::White) as f32);
    }

    #[test]
    fn symmetries_are_distinct_bijections() {
        for sym in 0..SYMMETRIES {
            let mut seen = [false; 64];
            for sq in 0..64 {
                let t = transform_square(sq, sym);
                assert!(!seen[t as usize]);
                seen[t as usize] = true;
            }
        }
        // a2（隅以外の非対称なマス）の行き先が 8 通りとも異なる
        let mut images: Vec<u8> = (0..SYMMETRIES).map(|s| transform_square(8, s)).collect();
        images.sort();
        images.dedup();
        assert_eq!(images.len(), 8);
    }

    #[test]
    fn transformed_board_keeps_legal_move_count_and_result() {
        let mut board = Board::new();
        for mv in [19, 18, 17] {
            let m = board.legal_moves();
            board = board.play(*m.iter().find(|x| **x == Move::Place(mv)).unwrap_or(&m[0])).unwrap();
        }
        for sym in 0..SYMMETRIES {
            let t = transform_board(&board, sym);
            assert_eq!(t.legal_mask().count_ones(), board.legal_mask().count_ones());
            assert_eq!(t.count(Color::Black), board.count(Color::Black));
            assert_eq!(t.legal_mask(), transform_bits(board.legal_mask(), sym));
        }
    }
}
