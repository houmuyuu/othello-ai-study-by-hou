//! 学習データ（局面と教師値）の保存・読み込み
//!
//! 教師値は「その局面の手番側から見た最終結果」（勝ち = 1, 引き分け = 0.5, 負け = 0）。
//! ファイル形式：マジック 8 バイト + 件数 (u64 LE) + 1 件 21 バイト（黒 u64, 白 u64, 手番 u8, 教師値 f32）。

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::board::{Board, Color};

const MAGIC: &[u8; 8] = b"OTHVAL1\n";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub board: Board,
    /// 手番側から見た最終結果（0〜1）
    pub target: f32,
}

pub fn write_samples(path: impl AsRef<Path>, samples: &[Sample]) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&(samples.len() as u64).to_le_bytes())?;
    for s in samples {
        w.write_all(&s.board.stones(Color::Black).to_le_bytes())?;
        w.write_all(&s.board.stones(Color::White).to_le_bytes())?;
        w.write_all(&[u8::from(s.board.turn() == Color::White)])?;
        w.write_all(&s.target.to_le_bytes())?;
    }
    w.flush()
}

pub fn read_samples(path: impl AsRef<Path>) -> io::Result<Vec<Sample>> {
    let path = path.as_ref();
    let invalid = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {msg}", path.display()));
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("学習データのファイルではありません"));
    }
    let mut u64_buf = [0u8; 8];
    r.read_exact(&mut u64_buf)?;
    let count = u64::from_le_bytes(u64_buf) as usize;

    let mut samples = Vec::with_capacity(count);
    let mut rec = [0u8; 21];
    for _ in 0..count {
        r.read_exact(&mut rec)?;
        let black = u64::from_le_bytes(rec[0..8].try_into().unwrap());
        let white = u64::from_le_bytes(rec[8..16].try_into().unwrap());
        let turn = if rec[16] == 0 { Color::Black } else { Color::White };
        let target = f32::from_le_bytes(rec[17..21].try_into().unwrap());
        if black & white != 0 || !(0.0..=1.0).contains(&target) {
            return Err(invalid("壊れたレコードがあります"));
        }
        samples.push(Sample { board: Board::from_bitboards(black, white, turn), target });
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Move;

    #[test]
    fn roundtrip() {
        let b0 = Board::new();
        let b1 = b0.play(Move::Place(19)).unwrap();
        let samples = vec![Sample { board: b0, target: 1.0 }, Sample { board: b1, target: 0.5 }];
        let path = std::env::temp_dir().join(format!("othello-data-test-{}.bin", std::process::id()));
        write_samples(&path, &samples).unwrap();
        assert_eq!(read_samples(&path).unwrap(), samples);
        let _ = std::fs::remove_file(path);
    }
}
