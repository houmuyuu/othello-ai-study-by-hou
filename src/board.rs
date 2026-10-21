//! オセロの盤面（ビットボード実装）
//!
//! マスのインデックスは `row * 8 + col`（a1 = 0, h8 = 63）。

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    Black,
    White,
}

impl Color {
    pub fn opponent(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }
}

/// 着手。`Pass` は合法手がないときのみ指せる。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Move {
    Place(u8),
    Pass,
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Move::Place(sq) => {
                let col = (b'a' + sq % 8) as char;
                let row = sq / 8 + 1;
                write!(f, "{col}{row}")
            }
            Move::Pass => write!(f, "pass"),
        }
    }
}

/// 対局結果（黒視点ではなく、色ごとの勝敗を表す）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Win(Color),
    Draw,
}

const NOT_A_FILE: u64 = 0xfefe_fefe_fefe_fefe;
const NOT_H_FILE: u64 = 0x7f7f_7f7f_7f7f_7f7f;

/// 8方向へのシフト（盤外へのはみ出しはマスクで除去）
const DIRECTIONS: [fn(u64) -> u64; 8] = [
    |b| b << 8,                // row+1
    |b| b >> 8,                // row-1
    |b| (b << 1) & NOT_A_FILE, // col+1
    |b| (b >> 1) & NOT_H_FILE, // col-1
    |b| (b << 9) & NOT_A_FILE, // row+1, col+1
    |b| (b << 7) & NOT_H_FILE, // row+1, col-1
    |b| (b >> 7) & NOT_A_FILE, // row-1, col+1
    |b| (b >> 9) & NOT_H_FILE, // row-1, col-1
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Board {
    black: u64,
    white: u64,
    turn: Color,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    /// 初期配置（d4/e5 = 白, d5/e4 = 黒, 黒番）
    pub fn new() -> Self {
        Board {
            black: (1 << 28) | (1 << 35),
            white: (1 << 27) | (1 << 36),
            turn: Color::Black,
        }
    }

    /// 石の配置と手番から盤面を作る（黒と白が重なっていたら panic）
    pub fn from_bitboards(black: u64, white: u64, turn: Color) -> Self {
        assert_eq!(black & white, 0, "black and white stones overlap");
        Board { black, white, turn }
    }

    /// 手番だけを変えた盤面（相手の着手可能数を数える場合など）
    pub fn with_turn(&self, turn: Color) -> Board {
        Board { turn, ..*self }
    }

    pub fn turn(&self) -> Color {
        self.turn
    }

    pub fn stones(&self, color: Color) -> u64 {
        match color {
            Color::Black => self.black,
            Color::White => self.white,
        }
    }

    pub fn count(&self, color: Color) -> u32 {
        self.stones(color).count_ones()
    }

    pub fn get(&self, sq: u8) -> Option<Color> {
        let bit = 1u64 << sq;
        if self.black & bit != 0 {
            Some(Color::Black)
        } else if self.white & bit != 0 {
            Some(Color::White)
        } else {
            None
        }
    }

    fn player_opponent(&self) -> (u64, u64) {
        match self.turn {
            Color::Black => (self.black, self.white),
            Color::White => (self.white, self.black),
        }
    }

    /// 手番側の合法手のビットマスク
    pub fn legal_mask(&self) -> u64 {
        let (p, o) = self.player_opponent();
        let empty = !(p | o);
        let mut moves = 0;
        for shift in DIRECTIONS {
            let mut t = shift(p) & o;
            for _ in 0..5 {
                t |= shift(t) & o;
            }
            moves |= shift(t) & empty;
        }
        moves
    }

    /// 手番側の合法手一覧。置ける場所がなく終局でもなければ `[Pass]`、終局なら空。
    pub fn legal_moves(&self) -> Vec<Move> {
        let mask = self.legal_mask();
        if mask != 0 {
            return BitIter(mask).map(Move::Place).collect();
        }
        if self.is_game_over() {
            Vec::new()
        } else {
            vec![Move::Pass]
        }
    }

    pub fn is_game_over(&self) -> bool {
        if self.legal_mask() != 0 {
            return false;
        }
        self.with_turn(self.turn.opponent()).legal_mask() == 0
    }

    pub fn result(&self) -> Option<GameResult> {
        if !self.is_game_over() {
            return None;
        }
        let (b, w) = (self.count(Color::Black), self.count(Color::White));
        Some(match b.cmp(&w) {
            std::cmp::Ordering::Greater => GameResult::Win(Color::Black),
            std::cmp::Ordering::Less => GameResult::Win(Color::White),
            std::cmp::Ordering::Equal => GameResult::Draw,
        })
    }

    fn flips(&self, sq: u8) -> u64 {
        let (p, o) = self.player_opponent();
        let mut flips = 0;
        for shift in DIRECTIONS {
            let mut line = 0;
            let mut m = shift(1u64 << sq);
            while m & o != 0 {
                line |= m;
                m = shift(m);
            }
            if m & p != 0 {
                flips |= line;
            }
        }
        flips
    }

    /// 着手して新しい盤面を返す。非合法手なら `None`。
    pub fn play(&self, mv: Move) -> Option<Board> {
        let mut next = *self;
        match mv {
            Move::Pass => {
                if self.legal_mask() != 0 || self.is_game_over() {
                    return None;
                }
            }
            Move::Place(sq) => {
                if sq >= 64 || self.legal_mask() & (1u64 << sq) == 0 {
                    return None;
                }
                let flips = self.flips(sq);
                let placed = flips | (1u64 << sq);
                match self.turn {
                    Color::Black => {
                        next.black |= placed;
                        next.white &= !flips;
                    }
                    Color::White => {
                        next.white |= placed;
                        next.black &= !flips;
                    }
                }
            }
        }
        next.turn = self.turn.opponent();
        Some(next)
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  a b c d e f g h")?;
        for row in 0..8 {
            write!(f, "{}", row + 1)?;
            for col in 0..8 {
                let c = match self.get(row * 8 + col) {
                    Some(Color::Black) => 'X',
                    Some(Color::White) => 'O',
                    None => '.',
                };
                write!(f, " {c}")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// ビットマスクの立っているビット位置を列挙する
pub struct BitIter(pub u64);

impl Iterator for BitIter {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        if self.0 == 0 {
            return None;
        }
        let sq = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1;
        Some(sq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_position_has_four_moves() {
        let board = Board::new();
        let mut moves: Vec<String> = board.legal_moves().iter().map(|m| m.to_string()).collect();
        moves.sort();
        assert_eq!(moves, ["c4", "d3", "e6", "f5"]);
    }

    #[test]
    fn play_flips_stones() {
        let board = Board::new().play(Move::Place(19)).unwrap(); // d3
        assert_eq!(board.count(Color::Black), 4);
        assert_eq!(board.count(Color::White), 1);
        assert_eq!(board.turn(), Color::White);
    }

    #[test]
    fn illegal_moves_are_rejected() {
        let board = Board::new();
        assert!(board.play(Move::Place(0)).is_none());
        assert!(board.play(Move::Pass).is_none());
    }
}
