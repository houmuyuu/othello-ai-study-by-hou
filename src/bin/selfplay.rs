//! AI 同士の自己対戦（MCTS vs Random）
//!
//! 使い方:
//!   cargo run --release --bin selfplay -- [--eval playout|positional] [--playouts N] [--iterations N | --time 秒] [--threads N] [--parallel tree|root] [--games N]

use std::sync::Arc;
use std::time::Duration;

use othello::ai::{
    Ai, Evaluator, Mcts, MctsConfig, ParallelMode, PlayoutEvaluator, PositionalEvaluator, RandomAi, SearchLimit,
};
use othello::board::{Board, Color, GameResult};

fn main() {
    let mut config = MctsConfig::default();
    let mut games = 2u32;
    let mut eval = "playout".to_string();
    let mut playouts = 1u32;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| usage(&format!("{arg} に値がありません")));
        match arg.as_str() {
            "--eval" => eval = value().clone(),
            "--playouts" => playouts = parse(value()),
            "--iterations" => config.limit = SearchLimit::Iterations(parse(value())),
            "--time" => config.limit = SearchLimit::Time(Duration::from_secs_f64(parse(value()))),
            "--threads" => config.threads = parse(value()),
            "--parallel" => {
                config.parallel = match value().as_str() {
                    "tree" => ParallelMode::Tree,
                    "root" => ParallelMode::Root,
                    v => usage(&format!("--parallel には tree か root を指定してください: {v}")),
                }
            }
            "--games" => games = parse(value()),
            _ => usage(&format!("不明な引数: {arg}")),
        }
    }

    let evaluator: Arc<dyn Evaluator> = match eval.as_str() {
        "playout" => Arc::new(PlayoutEvaluator::new(playouts)),
        "positional" => Arc::new(PositionalEvaluator::default()),
        v => usage(&format!("--eval には playout か positional を指定してください: {v}")),
    };

    let mut wins = 0;
    for game in 0..games {
        let mut mcts = Mcts::with_evaluator(config, evaluator.clone());
        let mut random = RandomAi::new();
        let mcts_color = if game % 2 == 0 { Color::Black } else { Color::White };

        let mut board = Board::new();
        while !board.is_game_over() {
            let mv = if board.turn() == mcts_color {
                let mv = mcts.select_move(&board).unwrap();
                let s = mcts.last_stats();
                if let Some(&(_, visits, value)) = s.children.first() {
                    println!(
                        "MCTS {mv}: {} iter / {} playouts / {:.2}s, visits={visits}, value={value:.3}",
                        s.iterations,
                        s.playouts,
                        s.elapsed.as_secs_f64()
                    );
                }
                mv
            } else {
                random.select_move(&board).unwrap()
            };
            board = board.play(mv).unwrap();
        }

        let result = board.result().unwrap();
        println!("{board}");
        println!(
            "Game {}: {} ({mcts_color:?}) vs {} -> 黒 {} : 白 {} ({result:?})\n",
            game + 1,
            mcts.name(),
            random.name(),
            board.count(Color::Black),
            board.count(Color::White)
        );
        if result == GameResult::Win(mcts_color) {
            wins += 1;
        }
    }
    println!("MCTS 勝利数: {wins}/{games}");
}

fn parse<T: std::str::FromStr>(s: &str) -> T {
    s.parse().unwrap_or_else(|_| usage(&format!("数値として解釈できません: {s}")))
}

fn usage(msg: &str) -> ! {
    eprintln!("{msg}");
    eprintln!("usage: othello [--eval playout|positional] [--playouts N] [--iterations N | --time SECS] [--threads N] [--parallel tree|root] [--games N]");
    std::process::exit(1);
}
