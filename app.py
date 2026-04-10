import os
import tempfile
from datetime import datetime
from typing import Optional

import streamlit as st
from markitdown import MarkItDown
from supabase import Client, create_client

# ページ設定
st.set_page_config(page_title="MarkItDown GUI", page_icon="📝", layout="wide")


# ============================================
# Supabase クライアント
# ============================================
@st.cache_resource
def get_supabase() -> Client:
    """Supabaseクライアントを生成してキャッシュ"""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def save_history(
    filename: str,
    file_size: int,
    file_type: str,
    markdown: str,
    storage_key: Optional[str] = None,
) -> None:
    """変換履歴をDBに1行挿入"""
    supabase = get_supabase()
    supabase.table("conversions").insert(
        {
            "filename": filename,
            "file_size": file_size,
            "file_type": file_type,
            "markdown": markdown,
            "char_count": len(markdown),
            "storage_key": storage_key,
        }
    ).execute()


def fetch_history(limit: int = 100) -> list[dict]:
    """変換履歴を新しい順で取得"""
    supabase = get_supabase()
    res = (
        supabase.table("conversions")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def upload_to_storage(file_bytes: bytes, filename: str) -> str:
    """元ファイルをStorageバケット 'uploads' に保存し、キー（パス）を返す"""
    supabase = get_supabase()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = f"{ts}_{filename}"
    supabase.storage.from_("uploads").upload(
        path=key,
        file=file_bytes,
        file_options={"upsert": "false"},
    )
    return key


# ============================================
# UI
# ============================================
st.title("MarkItDown GUI")
st.caption("ファイルをMarkdownに変換 + 変換履歴をSupabaseに保存")

tab_convert, tab_history = st.tabs(["変換", "履歴"])

# --------------------------------------------
# 変換タブ
# --------------------------------------------
with tab_convert:
    uploaded = st.file_uploader(
        "ファイルを選択（PDF / Word / Excel / PowerPoint / 画像 / 音声 など）",
        type=None,
    )
    save_original = st.checkbox(
        "元ファイルもStorageに保存する（Supabase Storage 上限1GBに注意）",
        value=False,
    )

    if uploaded is not None:
        suffix = os.path.splitext(uploaded.name)[1]
        file_bytes = uploaded.getvalue()

        # MarkItDownは拡張子からタイプを判定するため、一時ファイルに書き出して渡す
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            with st.spinner("変換中..."):
                result = MarkItDown().convert(tmp_path)
            st.success("変換が完了しました")

            # 元ファイルをStorageへ（任意）
            storage_key: Optional[str] = None
            if save_original:
                try:
                    with st.spinner("元ファイルをStorageに保存中..."):
                        storage_key = upload_to_storage(file_bytes, uploaded.name)
                except Exception as e:
                    st.warning(f"Storageへの保存に失敗しました: {e}")

            # DBに履歴保存
            try:
                save_history(
                    filename=uploaded.name,
                    file_size=len(file_bytes),
                    file_type=suffix.lstrip(".").lower() or "unknown",
                    markdown=result.text_content,
                    storage_key=storage_key,
                )
            except Exception as e:
                st.warning(f"履歴の保存に失敗しました: {e}")

            # ダウンロードボタン
            st.download_button(
                label="Markdownをダウンロード",
                data=result.text_content,
                file_name=os.path.splitext(uploaded.name)[0] + ".md",
                mime="text/markdown",
            )

            # プレビュー
            sub_raw, sub_rendered = st.tabs(["Markdown（生）", "プレビュー"])
            with sub_raw:
                st.text_area(
                    "Markdown",
                    result.text_content,
                    height=500,
                    label_visibility="collapsed",
                )
            with sub_rendered:
                st.markdown(result.text_content)

        except Exception as e:
            st.error(f"変換に失敗しました: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

# --------------------------------------------
# 履歴タブ
# --------------------------------------------
with tab_history:
    col_title, col_refresh = st.columns([4, 1])
    with col_title:
        st.subheader("変換履歴")
    with col_refresh:
        if st.button("更新", use_container_width=True):
            st.rerun()

    try:
        history = fetch_history(100)
    except Exception as e:
        st.error(f"履歴の取得に失敗しました: {e}")
        history = []

    if not history:
        st.info("まだ履歴がありません。変換タブでファイルをアップロードしてみてください。")
    else:
        # 表示用ラベル → 履歴アイテムの辞書
        options = {
            f"{h['filename']}  —  {h['created_at'][:19].replace('T', ' ')}": h
            for h in history
        }
        selected_label = st.selectbox(
            f"履歴（全{len(history)}件、新しい順）",
            list(options.keys()),
        )
        item = options[selected_label]

        # メタデータ表示
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("サイズ", f"{(item.get('file_size') or 0):,} B")
        c2.metric("拡張子", item.get("file_type") or "-")
        c3.metric("文字数", f"{(item.get('char_count') or 0):,}")
        c4.metric("日付", item["created_at"][:10])

        st.download_button(
            label="このMarkdownをダウンロード",
            data=item.get("markdown") or "",
            file_name=os.path.splitext(item["filename"])[0] + ".md",
            mime="text/markdown",
            key=f"dl_{item['id']}",
        )

        # 過去の変換結果をプレビュー
        h_raw, h_rendered = st.tabs(["Markdown（生）", "プレビュー"])
        with h_raw:
            st.text_area(
                "Markdown",
                item.get("markdown") or "",
                height=500,
                label_visibility="collapsed",
                key=f"ta_{item['id']}",
            )
        with h_rendered:
            st.markdown(item.get("markdown") or "")
