# AI Video Assistant

A Streamlit app that downloads audio from a YouTube video (or accepts an uploaded audio/video file), transcribes it locally with Whisper, and answers questions using transcript retrieval and Mistral.

## Run on Windows

Activate the virtual environment and start the app:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Alternatively, run without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Set `MISTRAL_API_KEY` in `.env`, or enter it in the app sidebar. Whisper runs locally; Mistral embeddings and chat require an internet connection and a valid API key. FFmpeg must be available on `PATH` for media decoding.

Choose a YouTube URL or upload a media file, select a Whisper model, then click **Transcribe video**. After transcription, ask questions in the chat. Answers are restricted to transcript excerpts and cite timestamps when available.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository and select `app.py` as the app entry point.
2. In the app's **Settings → Secrets**, add:

   ```toml
   MISTRAL_API_KEY = "your-mistral-api-key"
   ```

3. Keep `.env`, `.venv`, and `.streamlit/secrets.toml` out of Git. The repository's `.gitignore` excludes them. `packages.txt` installs FFmpeg, which Whisper needs to decode media.

The app reads the key from Streamlit secrets in the cloud and from `MISTRAL_API_KEY` in the environment (including `.env`) locally.
