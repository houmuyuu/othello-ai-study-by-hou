//! 価値ネットワークの強化学習ツール（GPU / CUDA）
//!
//! 使い方:
//!   cargo run --release --features cuda --bin rl -- loop [オプション]
//!   cargo run --release --features cuda --bin rl -- eval --model PATH [--vs playout|PATH] [オプション]
//!
//! loop の流れ（世代 k = 0, 1, 2, ...）:
//!   1. 自己対戦でデータ gen_k/data.bin を作る（k = 0 はプレイアウト評価の MCTS、以降は model_k の MCTS）
//!   2. 直近 window 世代分のデータで学習し、gen_{k+1}/model.bpk を作る（教師値は各局面の手番側から見た最終結果）
//!   3. 新しいモデルを、プレイアウト評価の MCTS と前世代のモデルと対戦させて強さを確認する
//!
//! 途中で止めても、既にあるファイルは再利用して続きから再開する。
//!
//! ネットワークの構成は `--arch small|resnet|resnet:CxB` で選ぶ（新しく学習を始めるときだけ有効。
//! 既定は resnet = 64 チャンネル × 6 ブロック）。続きから再開する場合は、保存済みモデルの構成を使う。
//! 構成の違うモデルどうしも `eval --model A --vs B` で対戦させて比較できる。
//!
//! 学習は常に GPU で行う。自己対戦・対戦での推論は既定で GPU（バッチ推論）だが、
//! `--selfplay-device cpu` で CPU（1 局面ずつ、CPU のコア数だけ並行）にもできる。
//! 小さなネットワークでは CPU のほうが速い場合がある。

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::tensor::Device;
use othello::ai::{Evaluator, PlayoutEvaluator};
use othello::nn::data::{Sample, read_samples, write_samples};
use othello::nn::selfplay::{MatchResult, SelfPlayConfig, play_match, self_play};
use othello::nn::train::{TrainConfig, train};
use othello::nn::{Architecture, BatchedNnEvaluator, NnEvaluator, ValueNet};

struct Options {
    dir: PathBuf,
    generations: usize,
    games: usize,
    iterations: u32,
    concurrency: usize,
    window: usize,
    eval_games: usize,
    train: TrainConfig,
    model: Option<PathBuf>,
    vs: String,
    /// loop で毎世代対戦させる比較用モデル（別の学習で作ったモデルなど）
    reference: Option<PathBuf>,
    /// 自己対戦・対戦での推論を GPU で行うか
    selfplay_on_gpu: bool,
    /// 新しく学習を始めるときのネットワーク構成（`None` なら既定の大きな構成）
    arch: Option<Architecture>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            dir: PathBuf::from("runs/default"),
            generations: 10,
            games: 1000,
            iterations: 200,
            concurrency: 256,
            window: 4,
            eval_games: 100,
            train: TrainConfig::default(),
            model: None,
            vs: "playout".to_string(),
            reference: None,
            selfplay_on_gpu: true,
            arch: None,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((command, rest)) = args.split_first() else { usage("サブコマンドを指定してください") };
    let options = parse_options(rest);
    let result = match command.as_str() {
        "loop" => run_loop(&options),
        "eval" => run_eval(&options),
        _ => usage(&format!("不明なサブコマンド: {command}")),
    };
    if let Err(e) = result {
        eprintln!("エラー: {e}");
        std::process::exit(1);
    }
}

fn parse_options(args: &[String]) -> Options {
    let mut o = Options::default();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| usage(&format!("{arg} に値がありません"))).as_str();
        match arg.as_str() {
            "--dir" => o.dir = PathBuf::from(value()),
            "--generations" => o.generations = parse(value()),
            "--games" => o.games = parse(value()),
            "--iterations" => o.iterations = parse(value()),
            "--concurrency" => o.concurrency = parse(value()),
            "--window" => o.window = parse::<usize>(value()).max(1),
            "--eval-games" => o.eval_games = parse(value()),
            "--epochs" => o.train.epochs = parse(value()),
            "--batch" => o.train.batch_size = parse(value()),
            "--lr" => o.train.learning_rate = parse(value()),
            "--model" => o.model = Some(PathBuf::from(value())),
            "--vs" => o.vs = value().to_string(),
            "--reference" => o.reference = Some(PathBuf::from(value())),
            "--arch" => {
                let v = value();
                o.arch = Some(Architecture::parse(v).unwrap_or_else(|| {
                    usage(&format!("--arch には small, resnet, resnet:CxB のいずれかを指定してください: {v}"))
                }))
            }
            "--selfplay-device" => {
                o.selfplay_on_gpu = match value() {
                    "gpu" => true,
                    "cpu" => false,
                    v => usage(&format!("--selfplay-device には gpu か cpu を指定してください: {v}")),
                }
            }
            _ => usage(&format!("不明な引数: {arg}")),
        }
    }
    o
}

fn parse<T: std::str::FromStr>(s: &str) -> T {
    s.parse().unwrap_or_else(|_| usage(&format!("数値として解釈できません: {s}")))
}

fn usage(msg: &str) -> ! {
    eprintln!("{msg}");
    eprintln!(
        "usage:
  rl loop [--dir DIR] [--generations N] [--games N] [--iterations N] [--concurrency N]
          [--window N] [--eval-games N] [--epochs N] [--batch N] [--lr F] [--selfplay-device gpu|cpu]
          [--arch small|resnet|resnet:CxB] [--reference PATH]
  rl eval --model PATH [--vs playout|PATH] [--eval-games N] [--iterations N] [--concurrency N]
          [--selfplay-device gpu|cpu]"
    );
    std::process::exit(2);
}

fn gen_dir(dir: &Path, k: usize) -> PathBuf {
    dir.join(format!("gen_{k:03}"))
}

fn model_path(dir: &Path, k: usize) -> PathBuf {
    gen_dir(dir, k).join("model.bpk")
}

fn data_path(dir: &Path, k: usize) -> PathBuf {
    gen_dir(dir, k).join("data.bin")
}

fn cpu_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// NN の評価関数
struct NnEval {
    evaluator: Arc<dyn Evaluator>,
    /// GPU のバッチ推論の場合、その統計を見るための参照
    batched: Option<Arc<BatchedNnEvaluator>>,
}

/// NN の評価関数を作る。GPU ならバッチ推論、CPU なら 1 局面ずつ推論する。
fn nn_evaluator(o: &Options, path: &Path) -> Result<NnEval, String> {
    let name = path.display().to_string();
    if o.selfplay_on_gpu {
        let device = Device::cuda(0);
        let model = ValueNet::load(path, &device)?;
        let e = Arc::new(BatchedNnEvaluator::new(model, device, o.concurrency, Duration::from_millis(2), name));
        Ok(NnEval { evaluator: e.clone(), batched: Some(e) })
    } else {
        let device = Device::flex();
        let model = ValueNet::load(path, &device)?;
        Ok(NnEval { evaluator: Arc::new(NnEvaluator::new(model, device, name)), batched: None })
    }
}

/// NN を使う自己対戦・対戦で同時に進める対局数
fn nn_concurrency(o: &Options) -> usize {
    if o.selfplay_on_gpu { o.concurrency } else { cpu_threads() }
}

/// 進捗を 1 行で上書き表示する
fn progress(label: &str, total: usize) -> impl Fn(usize) + Sync + '_ {
    let start = Instant::now();
    move |done| {
        if done == total || done % (total / 20).max(1) == 0 {
            eprint!("\r  {label}: {done}/{total} ({:.0} 秒)", start.elapsed().as_secs_f64());
            if done == total {
                eprintln!();
            }
            let _ = std::io::stderr().flush();
        }
    }
}

fn run_loop(o: &Options) -> Result<(), String> {
    std::fs::create_dir_all(&o.dir).map_err(|e| format!("{}: {e}", o.dir.display()))?;
    let log_path = o.dir.join("log.csv");
    if !log_path.exists() {
        std::fs::write(&log_path, "generation,samples,train_loss,val_loss,val_accuracy,vs_playout,vs_previous\n")
            .map_err(|e| e.to_string())?;
    }
    let train_device = Device::cuda(0).autodiff();

    // ネットワーク構成：学習済みのモデルがあればその構成で続ける
    let first_model = model_path(&o.dir, 1);
    let arch = if first_model.exists() {
        let existing = ValueNet::read_architecture(&first_model)?;
        if let Some(requested) = o.arch.filter(|a| *a != existing) {
            return Err(format!(
                "{} は構成 {existing} で学習済みです（--arch {requested} とは異なります）。別の --dir を指定してください",
                o.dir.display()
            ));
        }
        existing
    } else {
        o.arch.unwrap_or(Architecture::LARGE)
    };
    println!("構成: {arch}（パラメータ数 {}）", ValueNet::new(arch, &Device::flex()).num_params());

    for k in 0..o.generations {
        println!("== 世代 {k} ==");

        // 1. 自己対戦
        let data = data_path(&o.dir, k);
        if data.exists() {
            println!("  自己対戦データあり: {}", data.display());
        } else {
            let start = Instant::now();
            let (evaluator, gpu, concurrency): (Arc<dyn Evaluator>, _, _) = if k == 0 {
                (Arc::new(PlayoutEvaluator::new(1)), None, cpu_threads())
            } else {
                let nn = nn_evaluator(o, &model_path(&o.dir, k))?;
                (nn.evaluator, nn.batched, nn_concurrency(o))
            };
            let config = SelfPlayConfig {
                games: o.games,
                concurrency,
                iterations: o.iterations,
                seed: k as u64,
                ..Default::default()
            };
            let samples = self_play(evaluator, &config, progress("自己対戦", o.games));
            write_samples(&data, &samples).map_err(|e| e.to_string())?;
            println!("  {} 局面を生成（{:.0} 秒）", samples.len(), start.elapsed().as_secs_f64());
            if let Some(e) = &gpu {
                print_batch_stats(e, start.elapsed());
            }
        }

        // 2. 学習
        let next_model = model_path(&o.dir, k + 1);
        if next_model.exists() {
            println!("  学習済み: {}", next_model.display());
            continue;
        }
        let mut samples: Vec<Sample> = Vec::new();
        for j in k.saturating_sub(o.window - 1)..=k {
            samples.extend(read_samples(data_path(&o.dir, j)).map_err(|e| e.to_string())?);
        }
        let model = if k == 0 { ValueNet::new(arch, &train_device) } else { ValueNet::load(model_path(&o.dir, k), &train_device)? };
        println!("  学習: {} 局面（直近 {} 世代分）", samples.len(), o.window.min(k + 1));
        let start = Instant::now();
        let mut last = None;
        let config = TrainConfig { seed: k as u64, ..o.train };
        let model = train(model, &samples, &config, &train_device, |r| {
            println!(
                "    epoch {}: train_loss {:.4} / val_loss {:.4} / val_accuracy {:.1}%",
                r.epoch,
                r.train_loss,
                r.validation_loss,
                r.validation_accuracy * 100.0
            );
            last = Some(*r);
        });
        model.save(&next_model)?;
        model.save(o.dir.join("latest.bpk"))?;
        println!("  保存: {}（{:.0} 秒）", next_model.display(), start.elapsed().as_secs_f64());

        // 3. 強さの確認
        let match_config = SelfPlayConfig {
            games: o.eval_games,
            concurrency: nn_concurrency(o),
            iterations: o.iterations,
            seed: 1_000 + k as u64,
            ..Default::default()
        };
        let new_eval = nn_evaluator(o, &next_model)?.evaluator;
        let vs_playout = play_match(
            new_eval.clone(),
            Arc::new(PlayoutEvaluator::new(1)),
            &match_config,
            progress("対 プレイアウト", o.eval_games),
        );
        report("対 プレイアウト評価", &vs_playout);
        let vs_previous = if k == 0 {
            None
        } else {
            let prev = nn_evaluator(o, &model_path(&o.dir, k))?.evaluator;
            let r = play_match(new_eval.clone(), prev, &match_config, progress("対 前世代", o.eval_games));
            report("対 前世代", &r);
            Some(r)
        };
        // 比較用モデルとの対戦（結果は reference.csv に記録）
        if let Some(reference) = &o.reference {
            let ref_eval = nn_evaluator(o, reference)?.evaluator;
            let r = play_match(new_eval.clone(), ref_eval, &match_config, progress("対 比較用モデル", o.eval_games));
            report(&format!("対 {}", reference.display()), &r);
            let ref_log = o.dir.join("reference.csv");
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&ref_log).map_err(|e| e.to_string())?;
            if f.metadata().map_err(|e| e.to_string())?.len() == 0 {
                f.write_all(b"generation,reference,games,wins,losses,draws,score\n").map_err(|e| e.to_string())?;
            }
            let line = format!(
                "{},{},{},{},{},{},{:.4}\n",
                k + 1,
                reference.display(),
                o.eval_games,
                r.wins,
                r.losses,
                r.draws,
                r.score()
            );
            f.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
        }

        let last = last.expect("at least one epoch");
        let line = format!(
            "{},{},{:.5},{:.5},{:.4},{:.4},{}\n",
            k + 1,
            samples.len(),
            last.train_loss,
            last.validation_loss,
            last.validation_accuracy,
            vs_playout.score(),
            vs_previous.map_or(String::new(), |r| format!("{:.4}", r.score()))
        );
        let mut log = std::fs::OpenOptions::new().append(true).open(&log_path).map_err(|e| e.to_string())?;
        log.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
    }
    println!("完了。最新のモデル: {}", o.dir.join("latest.bpk").display());
    Ok(())
}

fn print_batch_stats(e: &BatchedNnEvaluator, elapsed: Duration) {
    use std::sync::atomic::Ordering;
    let batches = e.stats().batches.load(Ordering::Relaxed);
    let positions = e.stats().positions.load(Ordering::Relaxed);
    let inference = Duration::from_nanos(e.stats().inference_nanos.load(Ordering::Relaxed));
    println!(
        "  GPU 推論: {positions} 局面 / {batches} バッチ（平均 {:.1} 局面/バッチ, {:.0} 局面/秒, 推論 {:.2} ms/バッチ, 推論時間の割合 {:.0}%）",
        positions as f64 / batches.max(1) as f64,
        positions as f64 / elapsed.as_secs_f64(),
        inference.as_secs_f64() * 1000.0 / batches.max(1) as f64,
        inference.as_secs_f64() / elapsed.as_secs_f64() * 100.0
    );
}

fn report(label: &str, r: &MatchResult) {
    println!(
        "  {label}: {} 勝 {} 敗 {} 分（勝率 {:.1}%）",
        r.wins,
        r.losses,
        r.draws,
        r.score() * 100.0
    );
}

/// モデルの構成とパラメータ数
fn describe(path: &Path) -> Result<String, String> {
    let model = ValueNet::load(path, &Device::flex())?;
    Ok(format!("構成 {}（パラメータ数 {}）", model.architecture(), model.num_params()))
}

fn run_eval(o: &Options) -> Result<(), String> {
    let model = o.model.as_ref().unwrap_or_else(|| usage("--model を指定してください"));
    let a = nn_evaluator(o, model)?.evaluator;
    let b: Arc<dyn Evaluator> =
        if o.vs == "playout" { Arc::new(PlayoutEvaluator::new(1)) } else { nn_evaluator(o, Path::new(&o.vs))?.evaluator };
    let config = SelfPlayConfig {
        games: o.eval_games,
        concurrency: nn_concurrency(o),
        iterations: o.iterations,
        seed: 12_345,
        ..Default::default()
    };
    for path in std::iter::once(model.as_path()).chain((o.vs != "playout").then(|| Path::new(&o.vs))) {
        println!("  {}: {}", path.display(), describe(path)?);
    }
    let r = play_match(a, b, &config, progress("対戦", o.eval_games));
    report(&format!("{} 対 {}", model.display(), o.vs), &r);
    Ok(())
}
