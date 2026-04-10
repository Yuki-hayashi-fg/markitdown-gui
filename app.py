import os
import tempfile

import streamlit as st
from markitdown import MarkItDown

# ページ設定
st.set_page_config(page_title="MarkItDown GUI", page_icon="📝")

st.title("MarkItDown GUI")
st.caption("ファイルをアップロードしてMarkdownに変換します")

# ファイルアップローダー（拡張子は制限せず、MarkItDown側で対応判定）
uploaded = st.file_uploader(
    "ファイルを選択（PDF / Word / Excel / PowerPoint / 画像 / 音声 など）",
    type=None,
)

if uploaded is not None:
    # 一時ファイルに書き出してパスを MarkItDown に渡す
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    try:
        with st.spinner("変換中..."):
            result = MarkItDown().convert(tmp_path)

        st.success("変換が完了しました")

        # ダウンロードボタン
        st.download_button(
            label="Markdownをダウンロード",
            data=result.text_content,
            file_name=os.path.splitext(uploaded.name)[0] + ".md",
            mime="text/markdown",
        )

        # プレビュー（生テキスト と レンダリング結果の両方）
        tab_raw, tab_rendered = st.tabs(["Markdown（生）", "プレビュー"])
        with tab_raw:
            st.text_area(
                "生のMarkdown",
                result.text_content,
                height=500,
                label_visibility="collapsed",
            )
        with tab_rendered:
            st.markdown(result.text_content)

    except Exception as e:
        st.error(f"変換に失敗しました: {e}")
    finally:
        # 一時ファイルを削除
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
