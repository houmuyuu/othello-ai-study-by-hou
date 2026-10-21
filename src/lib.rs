pub mod ai;
pub mod board;
pub mod game;
pub mod gui;
pub mod nn;
pub mod player;
pub mod rng;

/// アプリのバージョン。リリースのビルドでは CI が環境変数 `OTHELLO_VERSION` で渡す（例：`0.1.3`）。
/// それ以外（ローカルでのビルド）では Cargo.toml のバージョンに `-dev` を付ける。
pub const VERSION: &str = match option_env!("OTHELLO_VERSION") {
    Some(v) => v,
    None => concat!(env!("CARGO_PKG_VERSION"), "-dev"),
};
