# Local Realtime Voice Assistant v3

Fully local voice assistant using:

- Windows Python 3.11
- Microphone input via `sounddevice`
- Silero VAD for speech start/end detection
- Faster-Whisper for STT
- Ollama for local LLM
- Piper for TTS
- Tool calling with:
  - `calculator`
  - `web_search`
  - `rag_search` over local documents using FAISS
- Wake word: `Hey assistant`

## Folder Structure

```text
ollama_voice_realtime_v3_repo/
├── app_realtime_v3.py
├── rag_ingest.py
├── rag_store.py
├── requirements.txt
├── .env.example
├── docs/
├── models/
│   ├── en_US-lessac-medium.onnx
│   └── en_US-lessac-medium.onnx.json
└── vectorstore/
```

## Important

This zip does **not** include large model files:

- Piper `.onnx` voice model
- Piper `.onnx.json` config
- Whisper model cache
- Ollama model files
- SentenceTransformer cache

Place your Piper files here:

```text
models/en_US-lessac-medium.onnx
models/en_US-lessac-medium.onnx.json
```

## Quick Start

```powershell
py -3.11 -m venv voice-env
.\voice-env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Start or verify Ollama:

```powershell
ollama list
ollama pull qwen2.5:3b
```

Put your PDFs/TXT/MD/DOCX files into `docs/`, then build the RAG index:

```powershell
python rag_ingest.py
```

Run the assistant:

```powershell
python app_realtime_v3.py
```

Say:

```text
Hey assistant search my documents for ACRES architecture
```

## Test Piper

```powershell
echo "hello world" | piper --model models\en_US-lessac-medium.onnx --output_file test.wav
start test.wav
```

## Test RAG

```powershell
python -c "from rag_store import rag_search; print(rag_search('architecture'))"
```
