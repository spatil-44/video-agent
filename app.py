from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import streamlit as st
import whisper
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from yt_dlp import YoutubeDL

load_dotenv()

SUPPORTED_MODELS = ("tiny", "base", "small", "medium")
ALLOWED_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def validate_youtube_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_YOUTUBE_HOSTS
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Enter a valid HTTPS link from YouTube or youtu.be.")
    return url.strip()


def format_transcript(result: dict[str, object]) -> str:
    segments = result.get("segments")
    if isinstance(segments, list) and segments:
        lines = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            start = int(float(segment.get("start", 0)))
            text = str(segment.get("text", "")).strip()
            if text:
                lines.append(f"[{start // 60:02d}:{start % 60:02d}] {text}")
        if lines:
            return "\n".join(lines)
    return str(result.get("text", "")).strip()


def chunk_transcript(transcript: str, size: int = 1200, overlap: int = 200) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap smaller than chunk size.")
    text = transcript.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


def retrieve_relevant_chunks(
    query_vector: list[float], chunks: list[str], vectors: list[list[float]], top_k: int = 5
) -> list[str]:
    if not chunks or len(chunks) != len(vectors):
        raise ValueError("Transcript chunks and their embeddings must have matching entries.")
    embeddings = np.asarray(vectors, dtype=np.float32)
    query_embedding = np.asarray(query_vector, dtype=np.float32)
    if (
        embeddings.ndim != 2
        or query_embedding.ndim != 1
        or embeddings.shape[1] != query_embedding.size
    ):
        raise ValueError("Embedding dimensions do not match.")
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    scores = np.divide(
        embeddings @ query_embedding,
        norms,
        out=np.zeros_like(norms),
        where=norms != 0,
    )
    selected = np.argsort(scores)[::-1][: min(top_k, len(chunks))]
    return [chunks[int(index)] for index in selected]


def get_mistral_api_key(override: str = "") -> str:
    if override.strip():
        return override.strip()
    environment_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        return str(st.secrets["MISTRAL_API_KEY"]).strip()
    except (KeyError, StreamlitSecretNotFoundError):
        return ""


@st.cache_resource(show_spinner="Loading Whisper transcription model…")
def load_whisper_model(model_name: str) -> whisper.Whisper:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported Whisper model: {model_name}")
    return whisper.load_model(model_name)


def download_youtube_audio(url: str, destination: Path) -> Path:
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(destination / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(validate_youtube_url(url), download=True)
        if not info:
            raise RuntimeError("YouTube did not return video information.")
        audio_path = Path(downloader.prepare_filename(info))
    if not audio_path.is_file():
        raise FileNotFoundError("The downloaded audio file could not be found.")
    return audio_path


def transcribe_audio(audio_path: Path, model_name: str) -> str:
    audio = whisper.load_audio(str(audio_path))
    if audio.size == 0:
        raise ValueError(
            "No audio samples could be decoded from this media. "
            "Check that the file is not empty and contains an audio track."
        )

    result = load_whisper_model(model_name).transcribe(audio, fp16=False)
    transcript = format_transcript(result)
    if not transcript:
        raise RuntimeError("Whisper did not produce a transcript for this media.")
    return transcript


def create_embeddings(api_key: str, chunks: list[str]) -> list[list[float]]:
    return MistralAIEmbeddings(api_key=api_key, model="mistral-embed").embed_documents(chunks)


def answer_question(
    api_key: str,
    question: str,
    transcript_chunks: list[str],
    transcript_vectors: list[list[float]],
    history: list[dict[str, str]],
) -> str:
    embeddings = MistralAIEmbeddings(api_key=api_key, model="mistral-embed")
    query_vector = embeddings.embed_query(question)
    relevant_chunks = retrieve_relevant_chunks(
        query_vector, transcript_chunks, transcript_vectors
    )

    messages = [
        SystemMessage(
            content=(
                "You are an assistant that answers questions about one video. "
                "Use only the transcript excerpts provided below. If they do not contain "
                "the answer, say so plainly. Cite supporting transcript timestamps in "
                "[MM:SS] format when available.\n\n"
                f"Relevant transcript excerpts:\n{chr(10).join(relevant_chunks)}"
            )
        )
    ]
    for item in history[-8:]:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    messages.append(HumanMessage(content=question))

    response = ChatMistralAI(
        api_key=api_key,
        model_name="mistral-small-latest",
        temperature=0.2,
    ).invoke(messages)
    if not isinstance(response.content, str):
        raise TypeError("Mistral returned a non-text response.")
    return response.content


def _transcribe_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, model_name: str) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.TemporaryDirectory() as temporary_directory:
        media_path = Path(temporary_directory) / f"upload{suffix}"
        media_path.write_bytes(uploaded_file.getvalue())
        return transcribe_audio(media_path, model_name)


def _queue_prompt(prompt: str) -> None:
    st.session_state["pending_prompt"] = prompt


def _clear_video() -> None:
    for key in ("transcript", "source_label", "transcript_chunks", "transcript_vectors", "messages"):
        st.session_state.pop(key, None)


def main() -> None:
    st.set_page_config(
        page_title="Framewise · Video assistant",
        page_icon=":material/smart_display:",
        layout="wide",
    )
    with st.sidebar:
        st.title("Framewise", anchor=False)
        st.caption("Your video knowledge workspace")
        st.subheader("Settings", icon=":material/tune:")
        model_name = st.selectbox(
            "Transcription model",
            SUPPORTED_MODELS,
            index=1,
            help="Larger models can be more accurate, but take longer and need more memory.",
        )
        api_key_override = st.text_input(
            "Mistral API key",
            type="password",
            placeholder="Use the configured Mistral key",
            help="Leave blank to use MISTRAL_API_KEY from your environment or Streamlit secrets.",
        )
        api_key = get_mistral_api_key(api_key_override)
        if api_key:
            st.badge("Mistral key ready", icon=":material/check_circle:", color="green")
        else:
            st.badge("API key needed for chat", icon=":material/key_off:", color="orange")
        st.caption("Transcription runs locally. Chat uses the Mistral API.")

    st.caption("VIDEO KNOWLEDGE STUDIO")
    st.title("Get more from every video", icon=":material/auto_awesome:")
    st.markdown(
        "Turn a video into a searchable conversation. Add a source, let Whisper "
        "transcribe it, then ask questions grounded in what was said."
    )

    transcript = st.session_state.get("transcript")
    if not transcript:
        with st.container(border=True):
            st.subheader("Add a video", icon=":material/video_library:")
            source_type = st.segmented_control(
                "Choose a source",
                ("YouTube link", "Upload a file"),
                default="YouTube link",
                key="source_type",
                label_visibility="collapsed",
                format_func=lambda source: (
                    f":material/{"link" if source == "YouTube link" else "upload"}: {source}"
                ),
            )
            with st.form("video_source_form"):
                youtube_url = ""
                uploaded_file = None
                if source_type == "YouTube link":
                    youtube_url = st.text_input(
                        "YouTube URL",
                        placeholder="Paste a YouTube link, e.g. https://youtu.be/…",
                    )
                else:
                    uploaded_file = st.file_uploader(
                        "Choose an audio or video file",
                        type=[
                            "mp3", "wav", "m4a", "aac", "ogg", "flac",
                            "mp4", "mov", "mkv", "webm",
                        ],
                        help="Supported formats include MP3, WAV, M4A, MP4, MOV, and WEBM.",
                    )
                process_video = st.form_submit_button(
                    "Transcribe video",
                    type="primary",
                    icon=":material/auto_awesome:",
                    width="stretch",
                )

        if process_video:
            if source_type == "YouTube link" and not youtube_url.strip():
                st.error("Paste a YouTube link to continue.")
            elif source_type == "Upload a file" and not uploaded_file:
                st.error("Choose a media file to continue.")
            else:
                try:
                    with st.spinner("Transcribing your video with Whisper…"):
                        if uploaded_file:
                            transcript = _transcribe_uploaded_file(uploaded_file, model_name)
                            source_label = uploaded_file.name
                        else:
                            with tempfile.TemporaryDirectory() as temporary_directory:
                                audio_path = download_youtube_audio(
                                    youtube_url, Path(temporary_directory)
                                )
                                transcript = transcribe_audio(audio_path, model_name)
                            source_label = youtube_url.strip()

                    st.session_state["transcript"] = transcript
                    st.session_state["source_label"] = source_label
                    st.session_state["transcript_chunks"] = chunk_transcript(transcript)
                    st.session_state["transcript_vectors"] = None
                    st.session_state["messages"] = []
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not process the media: {error}", icon=":material/error:")
                    with st.expander("Technical details"):
                        st.exception(error)

        st.markdown("#### What you can do")
        steps = st.columns(3)
        for column, icon, title, description in zip(
            steps,
            (":material/graphic_eq:", ":material/search:", ":material/chat:"),
            ("Transcribe", "Find the details", "Ask anything"),
            (
                "Whisper turns speech into a timestamped transcript.",
                "Relevant moments are retrieved from the transcript.",
                "Get concise answers with timestamps to check the source.",
            ),
        ):
            with column:
                with st.container(border=True, height="stretch"):
                    st.markdown(icon)
                    st.markdown(f"**{title}**")
                    st.caption(description)
    else:
        source_label = st.session_state["source_label"]
        header, reset = st.columns([5, 1], vertical_alignment="center")
        with header:
            st.badge("Transcript ready", icon=":material/check_circle:", color="green")
            st.subheader(source_label, anchor=False)
        with reset:
            st.button(
                "New video",
                icon=":material/add:",
                on_click=_clear_video,
                width="stretch",
            )

        if source_label.startswith(
            (
                "https://youtube.com/",
                "https://www.youtube.com/",
                "https://m.youtube.com/",
                "https://youtu.be/",
            )
        ):
            preview, summary = st.columns([1.25, 1], gap="large")
            with preview:
                st.video(source_label)
            with summary:
                st.markdown("#### Your video is ready")
                st.caption("Ask a question below to find answers in the transcript.")
                metric_columns = st.columns(2)
                metric_columns[0].metric("Transcript words", f"{len(transcript.split()):,}")
                metric_columns[1].metric(
                    "Search sections", f"{len(st.session_state['transcript_chunks']):,}"
                )
                st.download_button(
                    "Download transcript",
                    data=transcript,
                    file_name="video-transcript.txt",
                    mime="text/plain",
                    icon=":material/download:",
                    width="stretch",
                )
        else:
            info, actions = st.columns([3, 1], vertical_alignment="center")
            with info:
                st.caption(
                    f"{len(transcript.split()):,} words · "
                    f"{len(st.session_state['transcript_chunks']):,} searchable sections"
                )
            with actions:
                st.download_button(
                    "Download transcript",
                    data=transcript,
                    file_name="video-transcript.txt",
                    mime="text/plain",
                    icon=":material/download:",
                    width="stretch",
                )

        with st.expander("Read transcript", icon=":material/description:"):
            st.text_area(
                "Transcript",
                transcript,
                height=280,
                label_visibility="collapsed",
            )

        st.subheader("Ask about this video", icon=":material/forum:")
        messages = st.session_state.setdefault("messages", [])
        if not messages:
            st.caption("Answers are based on the transcript and include timestamps when available.")
            prompts = st.columns(3)
            for column, prompt in zip(
                prompts,
                (
                    "Give me a short summary",
                    "What are the key takeaways?",
                    "Find the most important moment",
                ),
            ):
                with column:
                    st.button(
                        prompt,
                        key=f"prompt_{prompt}",
                        on_click=_queue_prompt,
                        args=(prompt,),
                        width="stretch",
                    )
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                st.markdown("I’ve read the transcript. What would you like to know?")

        for message in messages:
            avatar = ":material/person:" if message["role"] == "user" else ":material/auto_awesome:"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        typed_question = st.chat_input("Ask a question about the video…")
        question = st.session_state.pop("pending_prompt", None) or typed_question
        if question:
            if not api_key:
                st.error(
                    "Add a Mistral API key in Settings or configure MISTRAL_API_KEY in .env.",
                    icon=":material/key_off:",
                )
            else:
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(question)
                try:
                    with st.spinner("Finding the right moments…"):
                        chunks = st.session_state["transcript_chunks"]
                        vectors = st.session_state.get("transcript_vectors")
                        if vectors is None:
                            vectors = create_embeddings(api_key, chunks)
                            st.session_state["transcript_vectors"] = vectors
                        answer = answer_question(api_key, question, chunks, vectors, messages)
                    messages.append({"role": "user", "content": question})
                    messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                        st.markdown(answer)
                except Exception as error:
                    st.error(f"Could not answer the question: {error}", icon=":material/error:")
                    with st.expander("Technical details"):
                        st.exception(error)


if __name__ == "__main__":
    main()
