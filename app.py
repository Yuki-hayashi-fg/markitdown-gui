import os
import re
import subprocess
import tempfile
from datetime import datetime
from typing import Optional

import streamlit as st
from markitdown import MarkItDown
from openai import OpenAI
from supabase import Client, create_client

# ページ設定
st.set_page_config(page_title="MarkItDown GUI", page_icon="📝", layout="wide")

# ============================================
# 定数
# ============================================
# 音声・動画ファイルの拡張子（これらは Whisper API で処理）
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}
AV_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

# Whisper API のファイルサイズ上限は 25MB。安全マージンを取って 24MB。
WHISPER_MAX_BYTES = 24 * 1024 * 1024

# 画像キャプション用に使うOpenAIモデル（ビジョン対応・低コスト）
VISION_MODEL = "gpt-4o-mini"


# ============================================
# Supabase クライアント
# ============================================
@st.cache_resource
def get_supabase() -> Client:
    """Supabaseクライアントを生成してキャッシュ"""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


# ============================================
# OpenAI クライアント
# ============================================
@st.cache_resource
def get_openai() -> OpenAI:
    """OpenAIクライアントを生成してキャッシュ"""
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ============================================
# MarkItDown クライアント
# ============================================
@st.cache_resource
def get_markitdown() -> MarkItDown:
    """MarkItDownインスタンスを生成。画像キャプション生成のためにOpenAIクライアントを渡す。"""
    return MarkItDown(llm_client=get_openai(), llm_model=VISION_MODEL)


# ============================================
# DB / Storage ヘルパー
# ============================================
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
# Whisper による音声・動画の文字起こし
# ============================================
def is_audio_or_video(filename: str) -> bool:
    """拡張子から音声・動画ファイルかを判定"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in AV_EXTENSIONS


def extract_audio_to_mp3(input_path: str) -> str:
    """ffmpeg で音声を抽出・圧縮して .mp3 に変換する。返り値は変換後のパス。"""
    output_path = input_path + ".extracted.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-y",              # 既存ファイルを上書き
            "-i", input_path,  # 入力
            "-vn",             # 映像を破棄
            "-ar", "16000",    # サンプリングレート 16kHz（Whisper推奨）
            "-ac", "1",        # モノラル
            "-b:a", "64k",     # ビットレート 64kbps
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def transcribe_with_whisper(file_path: str, original_name: str) -> str:
    """Whisper API で音声・動画を文字起こしし、Markdown文字列で返す"""
    client = get_openai()

    ext = os.path.splitext(file_path)[1].lower()
    path_to_send = file_path
    temp_audio: Optional[str] = None

    try:
        # 動画または大きいファイルは事前に ffmpeg で音声抽出・圧縮
        needs_extraction = (
            ext in VIDEO_EXTENSIONS
            or os.path.getsize(file_path) > WHISPER_MAX_BYTES
        )
        if needs_extraction:
            temp_audio = extract_audio_to_mp3(file_path)
            path_to_send = temp_audio

            # 抽出後もサイズオーバーの場合はエラー
            size_mb = os.path.getsize(path_to_send) / 1024 / 1024
            if os.path.getsize(path_to_send) > WHISPER_MAX_BYTES:
                raise RuntimeError(
                    f"音声抽出後のサイズが Whisper API 上限(25MB)を超えました: "
                    f"{size_mb:.1f}MB。より短い動画で試してください。"
                )

        with open(path_to_send, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ja",  # 日本語優先。英語中心なら "en"、自動検出なら引数削除
            )

        text = (transcript.text or "").strip()
        if not text:
            raise RuntimeError("文字起こし結果が空でした。音声が無音または認識できませんでした。")

        return f"# {original_name}（Whisper文字起こし）\n\n{text}\n"

    finally:
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.unlink(temp_audio)
            except OSError:
                pass


# ============================================
# 共通ユーティリティ
# ============================================
def safe_download_name(name: str) -> str:
    """ダウンロードファイル名として安全な文字列に変換"""
    stem = os.path.splitext(name)[0]
    stem = re.sub(r"https?://", "", stem)
    stem = re.sub(r"[^\w\-.]", "_", stem)
    stem = stem.strip("_") or "conversion"
    return stem[:100] + ".md"


def render_conversion_result(
    md_text: str,
    filename: str,
    file_size: int,
    file_type: str,
    original_bytes: Optional[bytes] = None,
    save_original: bool = False,
    converter_label: str = "MarkItDown",
) -> None:
    """変換結果を Storage/DB に保存して、ダウンロードボタンとプレビューを表示する共通処理"""
    st.success(f"変換が完了しました（{converter_label}）")

    # 元ファイルをStorageへ（任意・ファイルアップロード時のみ）
    storage_key: Optional[str] = None
    if save_original and original_bytes is not None:
        try:
            with st.spinner("元ファイルをStorageに保存中..."):
                storage_key = upload_to_storage(original_bytes, filename)
        except Exception as e:
            st.warning(f"Storageへの保存に失敗しました: {e}")

    # DBに履歴保存
    try:
        save_history(
            filename=filename,
            file_size=file_size,
            file_type=file_type,
            markdown=md_text,
            storage_key=storage_key,
        )
    except Exception as e:
        st.warning(f"履歴の保存に失敗しました: {e}")

    # ダウンロードボタン
    st.download_button(
        label="Markdownをダウンロード",
        data=md_text,
        file_name=safe_download_name(filename),
        mime="text/markdown",
    )

    # プレビュー
    sub_raw, sub_rendered = st.tabs(["Markdown（生）", "プレビュー"])
    with sub_raw:
        st.text_area(
            "Markdown",
            md_text,
            height=500,
            label_visibility="collapsed",
        )
    with sub_rendered:
        st.markdown(md_text)


# ============================================
# UI
# ============================================
st.title("MarkItDown GUI")
st.caption(
    "ドキュメント・画像は MarkItDown (+ OpenAI Vision)、"
    "音声・動画は Whisper、YouTube/Web は URL から直接変換。履歴はSupabaseに保存。"
)

tab_convert, tab_history = st.tabs(["変換", "履歴"])

# --------------------------------------------
# 変換タブ
# --------------------------------------------
with tab_convert:
    # 対応フォーマット一覧（折りたたみ）
    with st.expander("対応フォーマット詳細", expanded=False):
        st.markdown(
            """
| カテゴリ | 対応形式 | 変換品質 |
|---|---|---|
| **Word (.docx)** | 見出し・リスト・太字などの構造を保持 | ◎ 高品質 |
| **Excel (.xlsx / .xls / .csv)** | Markdownテーブルに変換 | ◎ 高品質 |
| **PowerPoint (.pptx)** | スライドごとにテキスト抽出 | ○ 良好 |
| **PDF** | テキスト抽出のみ。見出し・リスト等の書式は失われる | △ 限定的 |
| **HTML** | Markdownに変換 | ○ 良好 |
| **画像 (EXIF/OCR)** | メタデータ抽出。OCRやキャプション生成には外部LLMクライアントが必要 | △ 要LLM |
| **音声** | 文字起こし。外部LLMクライアントが必要 | △ 要LLM |
| **JSON / XML / CSV** | テキストベースなのでそのまま変換 | ○ 良好 |
| **ZIP** | 中身を展開して個別に変換 | ○ 良好 |
| **YouTube URL** | 字幕テキストを抽出 | ○ 良好 |

> このアプリでは画像キャプションに `gpt-4o-mini`、音声・動画の文字起こしに `whisper-1` を使用しています。
            """
        )

    input_mode = st.radio(
        "入力方式を選択",
        ["ファイル", "URL（YouTube / Web）"],
        horizontal=True,
    )

    # ============ ファイルアップロード ============
    if input_mode == "ファイル":
        uploaded = st.file_uploader(
            "ファイルを選択",
            type=None,
        )
        save_original = st.checkbox(
            "元ファイルもStorageに保存する（Supabase Storage 上限1GBに注意）",
            value=False,
        )

        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1]
            file_bytes = uploaded.getvalue()

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                # ファイルタイプに応じて変換方法を切り替え
                if is_audio_or_video(uploaded.name):
                    with st.spinner("Whisper APIで文字起こし中..."):
                        md_text = transcribe_with_whisper(tmp_path, uploaded.name)
                    converter_label = "Whisper"
                else:
                    with st.spinner("MarkItDownで変換中..."):
                        result = get_markitdown().convert(tmp_path)
                        md_text = result.text_content
                    converter_label = "MarkItDown"

                render_conversion_result(
                    md_text=md_text,
                    filename=uploaded.name,
                    file_size=len(file_bytes),
                    file_type=suffix.lstrip(".").lower() or "unknown",
                    original_bytes=file_bytes,
                    save_original=save_original,
                    converter_label=converter_label,
                )

            except subprocess.CalledProcessError as e:
                err = e.stderr.decode(errors="ignore")[:500] if e.stderr else str(e)
                st.error(f"ffmpeg 実行に失敗しました: {err}")
            except Exception as e:
                st.error(f"変換に失敗しました: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ============ URL入力 ============
    else:
        st.caption("YouTube動画（字幕抽出）、Wikipediaなどの一般Webページ（HTML→Markdown）")
        url = st.text_input(
            "URLを入力",
            placeholder="https://www.youtube.com/watch?v=... または https://example.com/...",
        )

        if st.button("変換", type="primary", disabled=not url):
            try:
                with st.spinner("URLの内容を取得・変換中..."):
                    result = get_markitdown().convert(url.strip())
                    md_text = result.text_content

                # URLからファイルタイプを推定
                url_lower = url.lower()
                if "youtube.com" in url_lower or "youtu.be" in url_lower:
                    file_type = "youtube"
                else:
                    file_type = "url"

                render_conversion_result(
                    md_text=md_text,
                    filename=url.strip(),
                    file_size=0,
                    file_type=file_type,
                    original_bytes=None,
                    save_original=False,
                    converter_label="MarkItDown (URL)",
                )

            except Exception as e:
                st.error(f"変換に失敗しました: {e}")

# --------------------------------------------
# 履歴タブ
# --------------------------------------------
with tab_history:
    col_title, col_refresh = st.columns([4, 1])
    with col_title:
        st.subheader("変換履歴（最新5件）")
    with col_refresh:
        if st.button("更新", use_container_width=True):
            st.rerun()

    try:
        history = fetch_history(5)
    except Exception as e:
        st.error(f"履歴の取得に失敗しました: {e}")
        history = []

    if not history:
        st.info("まだ履歴がありません。")
    else:
        for item in history:
            filename = item.get("filename") or ""
            display_name = filename if len(filename) <= 60 else filename[:60] + "…"
            executed_at = item["created_at"][:19].replace("T", " ")

            with st.expander(f"{display_name}   —   {executed_at}"):
                st.markdown(f"**入力**: `{filename}`")
                st.markdown("**出力**:")
                st.markdown(item.get("markdown") or "_（空）_")
                st.download_button(
                    label="ダウンロード",
                    data=item.get("markdown") or "",
                    file_name=safe_download_name(filename),
                    mime="text/markdown",
                    key=f"dl_{item['id']}",
                )
