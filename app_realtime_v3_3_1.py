import os
import re
import json
import sys
import time
import wave
import queue
import tempfile
import threading
import subprocess
import math
from dataclasses import dataclass

import numpy as np
import requests
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator

from rag_store import rag_search


@dataclass
class AppConfig:
    sample_rate: int = 16000
    channels: int = 1
    vad_window_samples: int = 512
    vad_threshold: float = 0.55
    min_silence_duration_ms: int = 350
    speech_pad_ms: int = 120
    min_utterance_ms: int = 650
    max_utterance_seconds: int = 45

    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"

    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_model: str = "qwen2.5:3b"
    ollama_temperature: float = 0.3

    piper_exe: str = "piper"
    piper_model: str = "models/en_US-lessac-medium.onnx"

    require_wake_word: bool = True
    wake_words: tuple = (
        "hey assistant",
        "hi assistant",
        "okay assistant",
        "ok assistant",
        "hello assistant",
        "hello ai",
    )

    smart_full_duplex: bool = True
    interrupt_phrases: tuple = (
        "stop",
        "cancel",
        "interrupt",
        "quiet",
        "be quiet",
        "pause",
        "hold on",
    )

    enable_text_input: bool = True
    typed_input_requires_wake_word: bool = False

    system_prompt: str = (
        "You are a concise local voice assistant with tool access. "
        "Use rag_search for local documents, PDFs, requirements, architecture notes, guides, and project files. "
        "Use calculator for math. Use web_search for current internet information. "
        "When answering from rag_search, base the answer only on retrieved context and briefly mention sources. "
        "If retrieved context is insufficient, say so clearly. Keep answers short and natural for voice."
    )


def calculator(expression: str) -> str:
    try:
        allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        return str(eval(expression, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"Calculation error: {e}"


# def web_search(query: str) -> str:
#     try:
#         response = requests.get(
#             "https://duckduckgo.com/html/",
#             params={"q": query},
#             timeout=15,
#             headers={"User-Agent": "Mozilla/5.0"},
#         )
#         if response.status_code != 200:
#             return f"Web search failed with status code {response.status_code}"
#         html = response.text
#         html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
#         html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
#         text = re.sub(r"(?s)<.*?>", " ", html)
#         text = re.sub(r"\s+", " ", text).strip()
#         return text[:3500]
#     except Exception as e:
#         return f"Web search error: {e}"

# def web_search(query: str) -> str:
#     # ---------- 1. Try DuckDuckGo JSON ----------
#     try:
#         r = requests.get(
#             "https://api.duckduckgo.com/",
#             params={
#                 "q": query,
#                 "format": "json",
#                 "no_html": 1
#             },
#             timeout=10,
#             headers={"User-Agent": "Mozilla/5.0"}
#         )

#         if r.status_code == 200:
#             data = r.json()

#             results = []
#             if data.get("AbstractText"):
#                 results.append(data["AbstractText"])

#             if data.get("RelatedTopics"):
#                 for item in data["RelatedTopics"][:5]:
#                     if isinstance(item, dict) and item.get("Text"):
#                         results.append(item["Text"])

#             if results:
#                 return " ".join(results)[:2000]

#     except Exception as e:
#         print("[web_search] DDG API failed:", e)

#     # ---------- 2. Fallback: DuckDuckGo HTML ----------
#     try:
#         r = requests.get(
#             "https://duckduckgo.com/html/",
#             params={"q": query},
#             timeout=10,
#             headers={"User-Agent": "Mozilla/5.0"}
#         )

#         if r.status_code == 200:
#             text = re.sub(r"(?s)<.*?>", " ", r.text)
#             text = re.sub(r"\s+", " ", text)
#             return text[:2000]

#     except Exception as e:
#         print("[web_search] DDG HTML failed:", e)

#     # ---------- 3. Fallback: Wikipedia ----------
#     try:
#         r = requests.get(
#             "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_"),
#             timeout=10
#         )

#         if r.status_code == 200:
#             data = r.json()
#             if "extract" in data:
#                 return data["extract"]

#     except Exception as e:
#         print("[web_search] Wikipedia failed:", e)

#     # ---------- FINAL ----------
#     return "Web search is currently unavailable due to network restrictions."


# def web_search(query: str) -> str:
#     try:
        
#         session = requests.Session()
#         session.trust_env = True 

#         r = session.get(
#             "https://duckduckgo.com/html/",
#             params={"q": query},
#             timeout=10,
#             headers={
#                 "User-Agent": "Mozilla/5.0"
#             }
#         )

#         if r.status_code != 200:
#             return f"Web search failed ({r.status_code})"

#         text = re.sub(r"(?s)<.*?>", " ", r.text)
#         text = re.sub(r"\s+", " ", text)

#         return text[:2000]

#     except Exception as e:
#         return f"Web search error: {e}"
    

def web_search(query: str) -> str:
    try:
        topic = query.split()[0]
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"

        print(f"[web_search] URL: {url}")

        r = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9"
            }
        )

        print(f"[web_search] Status: {r.status_code}")

        if r.status_code == 200:
            data = r.json()

            if "extract" in data:
                return data["extract"]

            return "No useful summary found."

        return f"Web search failed ({r.status_code})"

    except Exception as e:
        return f"Web search error: {e}"
    
    
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression safely.",
            "parameters": {
                "type": "object",
                "required": ["expression"],
                "properties": {"expression": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for current or external information.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Semantic search over local documents using FAISS.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        },
    },
]


class RealtimeVoiceAssistantV331:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.audio_q = queue.Queue()
        self.tts_q = queue.Queue()
        self.text_q = queue.Queue()
        self.running = True
        self.is_speaking = False
        self.interrupt_event = threading.Event()
        self.wake_armed = False
        self.messages = [{"role": "system", "content": self.cfg.system_prompt}]

        print("Loading Silero VAD...")
        torch.set_num_threads(1)
        try:
            self.vad_model = load_silero_vad(onnx=True)
        except Exception:
            self.vad_model = load_silero_vad()

        self.vad_iterator = VADIterator(
            self.vad_model,
            sampling_rate=self.cfg.sample_rate,
            threshold=self.cfg.vad_threshold,
            min_silence_duration_ms=self.cfg.min_silence_duration_ms,
            speech_pad_ms=self.cfg.speech_pad_ms,
        )

        print("Loading Whisper model...")
        self.whisper = WhisperModel(
            self.cfg.whisper_model,
            device=self.cfg.whisper_device,
            compute_type=self.cfg.whisper_compute_type,
        )

        threading.Thread(target=self.tts_worker, daemon=True).start()
        if self.cfg.enable_text_input:
            threading.Thread(target=self.text_input_worker, daemon=True).start()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"\nAudio status: {status}")
        self.audio_q.put(bytes(indata))

    def text_input_worker(self):
        while self.running:
            try:
                typed = input("\n⌨️  Type command: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if typed:
                self.text_q.put(typed)

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    def detect_wake_word(self, text: str):
        original = text.strip()
        normalized = self.normalize_text(original)
        for wake in self.cfg.wake_words:
            wake_norm = self.normalize_text(wake)
            if normalized == wake_norm:
                return True, ""
            if normalized.startswith(wake_norm + " "):
                pattern = re.compile(re.escape(wake), re.IGNORECASE)
                cleaned = pattern.sub("", original, count=1).strip(" ,.!?")
                return True, cleaned
        return False, ""

    def is_exit_command(self, text: str) -> bool:
        return self.normalize_text(text) in {"quit", "exit", "stop app", "goodbye"}

    def is_interrupt_only_command(self, text: str) -> bool:
        norm = self.normalize_text(text)
        return norm in {self.normalize_text(x) for x in self.cfg.interrupt_phrases}

    def should_accept_voice_during_tts(self, text: str):
        if self.is_interrupt_only_command(text):
            return True, "__interrupt_only__"
        wake, cmd = self.detect_wake_word(text)
        if wake:
            return True, cmd or "__interrupt_only__"
        return False, ""

    def should_process_after_wake(self, text: str, source: str):
        if source == "typed" and not self.cfg.typed_input_requires_wake_word:
            return True, text
        if not self.cfg.require_wake_word:
            return True, text
        if self.wake_armed:
            self.wake_armed = False
            return True, text
        wake, cmd = self.detect_wake_word(text)
        if not wake:
            print("[wake word not detected - ignored]")
            return False, ""
        if cmd:
            return True, cmd
        self.wake_armed = True
        self.tts_q.put("Yes?")
        print("[wake word detected - armed for next command]")
        return False, ""

    def clear_tts_queue(self):
        try:
            while True:
                self.tts_q.get_nowait()
                self.tts_q.task_done()
        except queue.Empty:
            pass

    def interrupt_tts(self, reason="interrupt"):
        if self.is_speaking:
            print(f"\n[interrupting speech: {reason}]")
            self.interrupt_event.set()
            self.clear_tts_queue()
            try:
                sd.stop()
            except Exception:
                pass
            if sys.platform.startswith("win"):
                try:
                    import winsound
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception:
                    pass

    def tts_worker(self):
        while self.running:
            try:
                text = self.tts_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if not text or not text.strip():
                self.tts_q.task_done()
                continue
            if self.interrupt_event.is_set():
                self.tts_q.task_done()
                continue
            try:
                self.speak_with_piper_interruptible(text)
            except Exception as e:
                print(f"\n[TTS ERROR] {e}")
            finally:
                self.tts_q.task_done()

    def get_wav_duration_seconds(self, wav_path: str) -> float:
        try:
            with wave.open(wav_path, "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return 0.0

    def play_wav_interruptible(self, wav_path: str):
        if sys.platform.startswith("win"):
            import winsound
            duration = self.get_wav_duration_seconds(wav_path)
            print(f"[TTS] Playing via winsound. Duration: {duration:.2f}s")
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            start = time.time()
            while self.running and not self.interrupt_event.is_set():
                if duration > 0 and time.time() - start >= duration:
                    break
                time.sleep(0.03)
            winsound.PlaySound(None, winsound.SND_PURGE)
            return

        import soundfile as sf
        data, sr = sf.read(wav_path, dtype="float32")
        sd.play(data, sr)
        sd.wait()

    def speak_with_piper_interruptible(self, text: str):
        print(f"\n[TTS] Queued: {text[:120]}")
        self.is_speaking = True
        self.interrupt_event.clear()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name
        try:
            print(f"[TTS] Temp WAV: {wav_path}")
            if not os.path.exists(self.cfg.piper_model):
                print(f"[TTS ERROR] Piper model not found: {self.cfg.piper_model}")
                return
            cmd = [self.cfg.piper_exe, "--model", self.cfg.piper_model, "--output_file", wav_path]
            print("[TTS] Running Piper...")
            try:
                result = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            except subprocess.TimeoutExpired:
                print("[TTS ERROR] Piper timed out.")
                return
            print(f"[TTS] Piper return code: {result.returncode}")
            if result.stderr:
                stderr_text = result.stderr.decode(errors="ignore").strip()
                if stderr_text:
                    print(f"[TTS] Piper stderr: {stderr_text[:500]}")
            if result.returncode != 0:
                print("[TTS ERROR] Piper failed.")
                return
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                print("[TTS ERROR] Piper did not create a valid WAV file.")
                return
            print(f"[TTS] WAV created. Size: {os.path.getsize(wav_path)} bytes")
            if self.interrupt_event.is_set():
                print("[TTS] Skipped playback because interrupt was already set.")
                return
            self.play_wav_interruptible(wav_path)
        finally:
            self.is_speaking = False
            self.interrupt_event.clear()
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def bytes_to_float_tensor(self, audio_bytes: bytes) -> torch.Tensor:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(audio_np)

    def save_wav_bytes(self, pcm_bytes: bytes, path: str):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.cfg.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.sample_rate)
            wf.writeframes(pcm_bytes)

    def transcribe_pcm_bytes(self, pcm_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name
        try:
            self.save_wav_bytes(pcm_bytes, wav_path)
            segments, _ = self.whisper.transcribe(wav_path, beam_size=1, vad_filter=False, language="en")
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def parse_tool_args(self, raw_args):
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except Exception:
                return {}
        return {}

    def ollama_chat(self, payload):
        r = requests.post(self.cfg.ollama_url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def ollama_with_tools(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        payload = {
            "model": self.cfg.ollama_model,
            "messages": self.messages,
            "tools": TOOLS,
            "stream": False,
            "options": {"temperature": self.cfg.ollama_temperature},
        }
        try:
            data = self.ollama_chat(payload)
        except Exception as e:
            return f"Sorry, I could not contact Ollama. Error: {e}"
        msg = data.get("message", {})
        self.messages.append(msg)
        calls = msg.get("tool_calls", [])
        if not calls:
            return msg.get("content", "").strip() or "I did not get a response."
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name")
            args = self.parse_tool_args(fn.get("arguments", {}))
            print(f"\n[Tool call] {name}: {args}")
            try:
                if name == "calculator":
                    result = calculator(args.get("expression", ""))
                elif name == "web_search":
                    result = web_search(args.get("query", ""))
                elif name == "rag_search":
                    result = rag_search(args.get("query", ""))
                else:
                    result = f"Unknown tool: {name}"
            except Exception as e:
                result = f"Tool execution error for {name}: {e}"
            print(f"[Tool result preview] {str(result)[:500]}")
            self.messages.append({"role": "tool", "name": name, "content": str(result)})
        try:
            final = self.ollama_chat({
                "model": self.cfg.ollama_model,
                "messages": self.messages,
                "stream": False,
                "options": {"temperature": self.cfg.ollama_temperature},
            })
        except Exception as e:
            return f"Tool completed, but final answer failed. Error: {e}"
        answer = final.get("message", {}).get("content", "").strip()
        if answer:
            self.messages.append({"role": "assistant", "content": answer})
        if len(self.messages) > 24:
            self.messages = [self.messages[0]] + self.messages[-22:]
        return answer or "I found context, but I could not produce a final answer."

    def process_user_text(self, text: str, source="voice") -> bool:
        if not text:
            return True
        print(f"You ({source}): {text}")
        if self.is_exit_command(text):
            print("Exiting.")
            return False
        if source == "typed" and self.is_speaking:
            self.interrupt_tts("typed command")
        if source == "voice" and self.is_speaking and self.cfg.smart_full_duplex:
            ok, cmd = self.should_accept_voice_during_tts(text)
            if not ok:
                print("[ignored during TTS: likely echo/noise/no wake word]")
                return True
            self.interrupt_tts("voice command during TTS")
            if cmd == "__interrupt_only__":
                print("[TTS stopped]")
                return True
            command = cmd
        else:
            ok, command = self.should_process_after_wake(text, source)
            if not ok:
                return True
        if self.is_exit_command(command):
            print("Exiting.")
            return False
        print(f"[command] {command}")
        answer = self.ollama_with_tools(command)
        print(f"AI: {answer}")
        self.tts_q.put(answer)
        return True

    def handle_pending_typed_input(self):
        while True:
            try:
                typed = self.text_q.get_nowait()
            except queue.Empty:
                break
            cont = self.process_user_text(typed, source="typed")
            self.text_q.task_done()
            if not cont:
                self.running = False
                break

    def run(self):
        print()
        print("===================================================")
        print(" Local Realtime Voice Assistant v3.3.1 Smart Full-Duplex")
        print(" Windows TTS playback uses winsound fallback")
        print(" Voice: Hey assistant <command>; during TTS say: stop / Hey assistant stop")
        print(" Keyboard: type a command and press Enter. Typed input interrupts TTS.")
        print(" Say/type: quit / exit / stop app / goodbye to quit.")
        print("===================================================")
        print()
        speech_buffer = bytearray()
        in_speech = False
        speech_started_at = None
        expected_bytes = self.cfg.vad_window_samples * self.cfg.channels * 2
        try:
            with sd.RawInputStream(
                samplerate=self.cfg.sample_rate,
                blocksize=self.cfg.vad_window_samples,
                dtype="int16",
                channels=self.cfg.channels,
                callback=self.audio_callback,
            ):
                while self.running:
                    if self.cfg.enable_text_input:
                        self.handle_pending_typed_input()
                        if not self.running:
                            break
                    try:
                        chunk = self.audio_q.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if len(chunk) != expected_bytes:
                        continue
                    speech_event = self.vad_iterator(self.bytes_to_float_tensor(chunk), return_seconds=False)
                    if speech_event:
                        if "start" in speech_event:
                            in_speech = True
                            speech_started_at = time.time()
                            speech_buffer = bytearray()
                            print("\n[start speech during TTS]" if self.is_speaking else "\n[start speech]")
                        if in_speech:
                            speech_buffer.extend(chunk)
                        if "end" in speech_event:
                            print("[end speech during TTS]" if self.is_speaking else "[end speech]")
                            in_speech = False
                            duration_ms = len(speech_buffer) / 2 / self.cfg.sample_rate * 1000
                            if duration_ms < self.cfg.min_utterance_ms:
                                print("[ignored short noise]")
                                speech_buffer = bytearray()
                                self.vad_iterator.reset_states()
                                continue
                            pcm = bytes(speech_buffer)
                            speech_buffer = bytearray()
                            print("Transcribing...")
                            user_text = self.transcribe_pcm_bytes(pcm)
                            if user_text:
                                cont = self.process_user_text(user_text, source="voice")
                                if not cont:
                                    self.running = False
                                    break
                            else:
                                print("[empty transcription]")
                            self.vad_iterator.reset_states()
                    else:
                        if in_speech:
                            speech_buffer.extend(chunk)
                            if speech_started_at and time.time() - speech_started_at > self.cfg.max_utterance_seconds:
                                print("[max utterance reached]")
                                in_speech = False
                                pcm = bytes(speech_buffer)
                                speech_buffer = bytearray()
                                print("Transcribing...")
                                user_text = self.transcribe_pcm_bytes(pcm)
                                if user_text:
                                    cont = self.process_user_text(user_text, source="voice")
                                    if not cont:
                                        self.running = False
                                        break
                                self.vad_iterator.reset_states()
        except KeyboardInterrupt:
            print("\nCtrl+C received. Exiting.")
        finally:
            self.running = False
            try:
                sd.stop()
            except Exception:
                pass
            if sys.platform.startswith("win"):
                try:
                    import winsound
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception:
                    pass


if __name__ == "__main__":
    config = AppConfig(
        ollama_model="qwen2.5:3b",
        piper_model="models/en_US-lessac-medium.onnx",
        piper_exe="piper",
        whisper_model="base",
        whisper_compute_type="int8",
        whisper_device="cpu",
        require_wake_word=True,
        smart_full_duplex=True,
        enable_text_input=True,
        typed_input_requires_wake_word=False,
    )
    app = RealtimeVoiceAssistantV331(config)
    app.run()
