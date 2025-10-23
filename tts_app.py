import os
from pathlib import Path
from typing import List
import uuid

import streamlit as st
from openai import OpenAI
from pydub import AudioSegment

# -----------------------
# Config
# -----------------------
DEFAULT_MODEL = "gpt-4o-mini-tts"
ALLOWED_VOICES = [
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "coral", "verse", "ballad", "ash", "sage", "marin", "cedar"
]
MAX_CHARS_PER_CHUNK = 4000  # conservative; API can handle more, but this is safe for TTS
OUTPUT_DIR = Path("tts_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def chunk_text(text: str, max_len: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """Split text into chunks on paragraph/period boundaries to aid natural TTS."""
    text = text.replace("\r\n", "\n").strip()
    if len(text) <= max_len:
        return [text]

    chunks, current = [], []
    current_len = 0

    # Split by paragraphs first for better prosody
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_len:
            # Split overly long paragraph by sentences
            sentences = []
            buf = []
            for piece in para.replace("?", "?.").replace("!", "!.").split("."):
                piece = piece.strip()
                if not piece:
                    continue
                buf.append(piece)
                if len(" ".join(buf)) > max_len * 0.8:  # soft boundary
                    sentences.append(". ".join(buf) + ".")
                    buf = []
            if buf:
                sentences.append(". ".join(buf) + ".")
            units = sentences
        else:
            units = [para]

        for unit in units:
            u = unit if unit.endswith((".", "!", "?", "…")) else unit + "."
            if current_len + len(u) + 1 <= max_len:
                current.append(u)
                current_len += len(u) + 1
            else:
                chunks.append("\n\n".join(current))
                current = [u]
                current_len = len(u) + 1

    if current:
        chunks.append("\n\n".join(current))
    return chunks

def synthesize_chunk(client: OpenAI, model: str, voice: str, text: str, outfile: Path):
    resp = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    outfile.write_bytes(resp.content)

def stitch_mp3(parts: List[Path], out_file: Path):
    """Concatenate mp3 parts into a single mp3 using pydub (ffmpeg required)."""
    combined = None
    for i, p in enumerate(parts):
        seg = AudioSegment.from_file(p, format="mp3")
        combined = seg if combined is None else combined + seg
        # small crossfade/pause between chunks for natural flow (optional)
        if i < len(parts) - 1:
            combined += AudioSegment.silent(duration=200)  # 0.2s
    combined.export(out_file, format="mp3")

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="🎙️ Text → Speech (OpenAI TTS)", page_icon="🎧", layout="centered")
st.title("🎙️ Text-to-Speech Generator")
st.caption("Paste text or upload a .txt file, choose a voice, and download the MP3.")

with st.sidebar:
    st.markdown("### Settings")
    model = st.text_input("Model", value=DEFAULT_MODEL, help="Recommended: gpt-4o-mini-tts")
    voice = st.selectbox("Voice", ALLOWED_VOICES, index=ALLOWED_VOICES.index("alloy"))
    base_filename = st.text_input("Base file name", value="tts_output")
    keep_parts = st.checkbox("Keep chunk files", value=False, help="Save individual chunk mp3s for inspection")
    st.markdown("---")
    st.markdown("**Tip**: Long text is auto-chunked so you can synthesize book chapters without hitting limits.")

api_key_ok = bool(os.getenv("OPENAI_API_KEY"))
if not api_key_ok:
    st.warning("OPENAI_API_KEY is not set in your environment. Set it and restart the app.", icon="⚠️")

tab1, tab2 = st.tabs(["📝 Paste Text", "📄 Upload .txt"])
with tab1:
    text_input = st.text_area("Paste your text here:", height=260, placeholder="Paste any text...")

with tab2:
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    upload_text = ""
    if uploaded is not None:
        upload_text = uploaded.read().decode("utf-8", errors="ignore")

# Merge inputs: uploaded text takes priority if present
final_text = upload_text.strip() if upload_text else text_input.strip()

colA, colB = st.columns(2)
with colA:
    est_chars = len(final_text)
    st.write(f"**Characters:** {est_chars:,}")
with colB:
    # rough words per minute to estimate duration
    wpm = 135
    est_mins = max(1, int((len(final_text.split()) / wpm) + 0.5)) if final_text else 0
    st.write(f"**Est. duration:** {est_mins} min")

generate = st.button("🔊 Generate Audio", type="primary", disabled=not final_text or not api_key_ok)

# -----------------------
# Run
# -----------------------
if generate:
    try:
        client = OpenAI()  # uses env var
        chunks = chunk_text(final_text, MAX_CHARS_PER_CHUNK)
        st.info(f"Processing {len(chunks)} chunk(s) …", icon="ℹ️")

        session_id = uuid.uuid4().hex[:8]
        part_files = []
        prog = st.progress(0, text="Synthesizing…")

        for i, chunk in enumerate(chunks, start=1):
            part_path = OUTPUT_DIR / f"{base_filename}_{voice}_part{i:02d}_{session_id}.mp3"
            synthesize_chunk(client, model, voice, chunk, part_path)
            part_files.append(part_path)
            prog.progress(i / len(chunks), text=f"Chunk {i}/{len(chunks)}")

        # Stitch into a single file
        final_out = OUTPUT_DIR / f"{base_filename}_{voice}_{session_id}.mp3"
        stitch_mp3(part_files, final_out)

        st.success("✅ Audio ready!")
        st.audio(str(final_out))
        with open(final_out, "rb") as f:
            st.download_button(
                label="⬇️ Download MP3",
                data=f,
                file_name=final_out.name,
                mime="audio/mpeg"
            )

        if keep_parts:
            st.write("Chunk files:")
            for pf in part_files:
                with open(pf, "rb") as f:
                    st.download_button(f"Download {pf.name}", f, file_name=pf.name, mime="audio/mpeg")
        else:
            # Clean up chunk parts
            for pf in part_files:
                try:
                    pf.unlink(missing_ok=True)
                except Exception:
                    pass

    except Exception as e:
        st.error(f"Generation failed: {e}")
