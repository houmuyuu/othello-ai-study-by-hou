# オセロ AI

ゲーム AI の**学習のため**のプロジェクトです。
オセロを題材に、モンテカルロ木探索（MCTS）やニューラルネットによる評価関数など、
ゲーム AI のアルゴリズムを実装して試しています。

オセロの AI が実装されていて、**AI と対戦**したり、**AI 同士を対戦**させたりできます。

## 実装している AI

| AI | 内容 |
|---|---|
| ランダム | 合法手からランダムに打つ |
| MCTS | モンテカルロ木探索。葉の評価方法を次から選べる |
| 　└ プレイアウト | 終局までランダムに打った結果で評価する |
| 　└ 静的評価 | マスの重みと着手可能数で評価する |
| 　└ NN | 自己対戦で強化学習した価値ネットワークで評価する（既定は同梱の学習済みモデル） |

NN の推論は、NVIDIA の GPU が使えれば GPU、使えなければ CPU で行います。

## 遊び方

### ダウンロード（Windows）

[Releases](../../releases) から最新の `othello-windows-x64.zip` をダウンロードして展開し、`othello.exe` を起動します。

- NVIDIA の GPU がある場合は、ドライバが入っていれば GPU で動きます（CUDA Toolkit は不要です）
- GPU がない PC でも、CPU で動きます

### ソースから実行

[Rust](https://www.rust-lang.org/) をインストールして、次を実行します。

```sh
cargo run --release
```

NVIDIA の GPU を使う場合は `cargo run --release --features cuda` で起動します。

## 操作方法

画面の左が設定、中央が盤面、右が対局の情報です。

1. 左の「先手（黒）」「後手（白）」で、それぞれ「人間」「ランダム」「MCTS」を選びます
   - AI と対戦する：片方を「人間」、もう片方を「MCTS」にします
   - AI 同士を対戦させる：両方を AI にします
2. 「開始」を押すと対局が始まります
3. 自分の番では、盤面の打てるマスが示されるので、クリックして石を置きます（打てる場所がないときは自動でパスします）
4. 「一時停止」「再開」「リセット」で対局を止めたり、やり直したりできます

MCTS では思考時間（既定は 1 手 5 秒）や評価関数などを設定できます。
右側には、AI が考えた手の勝率や探索回数、推論に使っているデバイス（GPU / CPU）が表示されます。

## ライセンス

このプロジェクトのソースコードは、次のどちらかを選んで利用できます（デュアルライセンス）。

- [MIT License](LICENSE-MIT)
- [Apache License 2.0](LICENSE-APACHE)

配布用の zip には、次のものも含まれています。これらはこのプロジェクトのライセンスの対象外で、それぞれのライセンスに従います。

- `nvrtc64_*.dll`、`nvrtc-builtins64_*.dll`、`cuda/nvvm/libdevice/libdevice.10.bc`：NVIDIA の CUDA Toolkit の再配布可能なファイル（NVRTC と libdevice）。NVIDIA のライセンス（同梱の `NVIDIA_CUDA_LICENSE.txt`）に従います
- `othello.exe` に組み込まれているライブラリ：それぞれのライセンスに従います（一覧とライセンス文は同梱の `THIRD_PARTY_LICENSES.html`、Apache License 2.0 で同梱が必要な NOTICE は `THIRD_PARTY_NOTICES.txt`）
