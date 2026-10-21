//! panic の内容の記録
//!
//! 配布版の exe はコンソールを持たないため、AI の思考スレッドなどが panic しても内容が見えない。
//! panic フックで最初の panic の内容を残しておき、画面に表示できるようにする。
//! （探索スレッドの panic は呼び出し元のスレッドにも伝わって panic が連鎖するため、原因を示す最初の 1 件を残す）

use std::panic::PanicHookInfo;
use std::sync::Mutex;

static FIRST: Mutex<Option<String>> = Mutex::new(None);

/// panic フックを設定する（既存のフックも引き続き呼ぶ）
pub fn install() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        record(&FIRST, describe(info));
        previous(info);
    }));
}

/// 記録した panic の内容を取り出す（取り出すと空になる）
pub fn take() -> Option<String> {
    FIRST.lock().ok()?.take()
}

/// まだ何も記録されていなければ記録する（連鎖した後続の panic で上書きしない）
fn record(slot: &Mutex<Option<String>>, message: String) {
    if let Ok(mut first) = slot.lock() {
        first.get_or_insert(message);
    }
}

fn describe(info: &PanicHookInfo) -> String {
    let payload = info.payload();
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "不明なエラー".to_string());
    let location = info.location().map(|l| format!("（{}:{}）", l.file(), l.line())).unwrap_or_default();
    format!("{message}{location}")
}

#[cfg(test)]
mod tests {
    use super::*;

    // panic フックはプロセス全体で共有され、並行して動く他のテストの panic も記録されるため、
    // フックそのものではなく、記録の規則を専用の記録先で確かめる
    #[test]
    fn only_first_message_is_kept_until_taken() {
        let slot = Mutex::new(None);
        record(&slot, "最初".to_string());
        record(&slot, "連鎖".to_string());
        assert_eq!(slot.lock().unwrap().take().as_deref(), Some("最初"));
        record(&slot, "次".to_string());
        assert_eq!(slot.lock().unwrap().take().as_deref(), Some("次"));
    }
}
