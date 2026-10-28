import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from app import (
    chunk_transcript,
    format_transcript,
    get_mistral_api_key,
    retrieve_relevant_chunks,
    transcribe_audio,
    validate_youtube_url,
)


class TranscriptHelpersTests(unittest.TestCase):
    def test_validate_youtube_url_accepts_supported_https_hosts(self) -> None:
        self.assertEqual(
            validate_youtube_url(" https://youtu.be/example "),
            "https://youtu.be/example",
        )

    def test_validate_youtube_url_rejects_non_youtube_and_non_https_urls(self) -> None:
        for url in ("http://youtu.be/example", "https://example.com/video"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_youtube_url(url)

    def test_format_transcript_adds_segment_timestamps(self) -> None:
        transcript = format_transcript(
            {"segments": [{"start": 62.4, "text": " A useful point."}]}
        )
        self.assertEqual(transcript, "[01:02] A useful point.")

    def test_chunk_transcript_preserves_text_across_overlaps(self) -> None:
        transcript = "A" * 50
        chunks = chunk_transcript(transcript, size=20, overlap=5)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "A" * 20)
        self.assertEqual(chunks[-1], "A" * 20)

    def test_retrieve_relevant_chunks_returns_highest_cosine_similarity(self) -> None:
        chunks = ["first", "second"]
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        result = retrieve_relevant_chunks([1.0, 0.1], chunks, vectors)
        self.assertEqual(result[0], "first")

    @patch("app.st.secrets", {"MISTRAL_API_KEY": "cloud-key"})
    @patch("app.os.getenv", return_value="")
    def test_get_mistral_api_key_reads_streamlit_secrets(
        self, get_env: Mock
    ) -> None:
        self.assertEqual(get_mistral_api_key(), "cloud-key")

    @patch("app.st.secrets", {})
    @patch("app.os.getenv", return_value="environment-key")
    def test_get_mistral_api_key_prefers_environment(
        self, get_env: Mock
    ) -> None:
        self.assertEqual(get_mistral_api_key(), "environment-key")

    def test_get_mistral_api_key_prefers_nonempty_override(self) -> None:
        self.assertEqual(get_mistral_api_key(" override-key "), "override-key")

    @patch("app.load_whisper_model")
    @patch("app.whisper.load_audio", return_value=np.array([], dtype=np.float32))
    def test_transcribe_audio_rejects_media_without_decoded_samples(
        self, load_audio: Mock, load_model: Mock
    ) -> None:
        with self.assertRaisesRegex(ValueError, "No audio samples could be decoded"):
            transcribe_audio(Path("empty.wav"), "base")

        load_audio.assert_called_once_with("empty.wav")
        load_model.assert_not_called()

    @patch("app.whisper.load_audio")
    @patch("app.load_whisper_model")
    def test_transcribe_audio_passes_decoded_samples_to_whisper(
        self, load_model: Mock, load_audio: Mock
    ) -> None:
        audio = np.ones(16000, dtype=np.float32)
        load_audio.return_value = audio
        model = Mock()
        model.transcribe.return_value = {
            "segments": [{"start": 0, "text": " Hello"}]
        }
        load_model.return_value = model

        transcript = transcribe_audio(Path("audio.wav"), "base")

        self.assertEqual(transcript, "[00:00] Hello")
        load_audio.assert_called_once_with("audio.wav")
        load_model.assert_called_once_with("base")
        model.transcribe.assert_called_once_with(audio, fp16=False)


if __name__ == "__main__":
    unittest.main()
