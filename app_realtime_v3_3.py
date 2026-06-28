import os
import re
import json
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
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator

from rag_store import rag_search


# ============================================================
# Local Realtime Voice Assistant v3.3 Smart Full-Duplex
# ============================================================
# What v3.3 changes vs v3.2:
# - Mic keeps listening while Piper speaks.
# - It does NOT immediately interrupt on any VAD noise.
# - During TTS, speech is recorded + transcribed first.
# - Only a real command interrupts TTS:
#     - voice starts with wake word, e.g. "Hey assistant stop"
#     - voice is an explicit interrupt phrase, e.g. "stop", "cancel", "interrupt"
#     - typed input always interrupts immediately
# - This avoids the common speaker-echo problem where the assistant cancels itself.
#
# Recommended:
# - Best full-duplex experience: headphones.
# - If using speakers, keep require_wake_word=True.
# ============================================================


@dataclass
class AppConfig:
    # Audio / VAD
    sample_rate: int = 16000
    channels: int = 1
    vad_window_samples: int = 512
    vad_threshold: float = 0.55
    min_silence_duration_ms: int = 350
    speech_pad_ms: int = 120
    min_utterance_ms: int = 650
    max_utterance_seconds: int = 45

    # Whisper
    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"

    # Ollama
    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_model: str = "qwen2.5:3b"
    ollama_temperature: float = 0.3

    # Piper
    piper_exe: str = "piper"
    piper_model: str = "models/en_US-lessac-medium.onnx"

    # Conversation behaviour
    require_wake_word: bool = True
    wake_words: tuple = (
        "hey assistant",
        "hi assistant",
        "okay assistant",
        "ok assistant",
        "hello assistant",
        "hello ai",
    )

    # Smart full duplex behaviour
    smart_full_duplex: bool = True
    allow_voice_interrupt_during_tts: bool = True
    interrupt_phrases: tuple = (
        "stop",
        "cancel",
        "interrupt",
        "quiet",
        "be quiet",
        "pause",
        "hold on",
    )

    # Typed input behaviour
    enable_text_input: bool = True
    typed_input_requires_wake_word: bool = False

    # TTS playback
    playback_chunk_samples: int = 2048

    system_prompt: str = (
        "You are a concise local voice assistant with tool access. "
        "Use rag_search for local documents, PDFs, requirements, architecture notes, guides, and project files. "
        "Use calculator for math. "
        "Use web_search for current internet information. "
        "When answering from rag_search, base the answer only on retrieved context and briefly mention sources. "
        "If retrieved context is insufficient, say so clearly. "
        "Keep answers short and natural for voice."
    )


# ============================================================
# Tool functions
# ============================================================


def calculator(expression: str) -> str:
    try:
        allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        return str(eval(expression, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"Calculation error: {e}"


def web_search(query: str) -> str:
    try:
        response = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if response.status_code != 200:
            return f"Web search failed with status code {response.status_code}"

        html = response.text
        html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
        text = re.sub(r"(?s)<.*?>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:3500]
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
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, for example: 45 * 23",
                    }
                },
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
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Internet search query",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Semantic search over local documents using FAISS. "
                "Use for local PDFs, TXT, Markdown, DOCX, requirements, architecture notes, guides, and project documents."
            ),
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic local document query",
                    }
                },
            },
        },
    },
]


class RealtimeVoiceAssistantV33:
    def __init__(self, config: AppConfig):
        self.cfg = config
        self.audio_q = queue.Queue()
        self.tts_q = queue.Queue()
        self.text_q = queue.Queue()

        self.running = True
        self.is_speaking = False
        self.is_processing = False
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

        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()

        if self.cfg.enable_text_input:
            self.text_thread = threading.Thread(target=self.text_input_worker, daemon=True)
            self.text_thread.start()

    # ------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------

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

    def bytes_to_float_tensor(self, audio_bytes: bytes) -> torch.Tensor:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(audio_np)

    def save_wav_bytes(self, pcm_bytes: bytes, path: str):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.cfg.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.sample_rate)
            wf.writeframes(pcm_bytes)

    # ------------------------------------------------------------
    # Text / wake helpers
    # ------------------------------------------------------------

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

    def should_accept_voice_during_tts(self, user_text: str):
        """
        During Piper playback, do not interrupt on random noise/echo.
        Accept only:
        - explicit interrupt words: stop/cancel/etc.
        - wake-word command: Hey assistant <command>
        """
        if self.is_interrupt_only_command(user_text):
            return True, "__interrupt_only__"

        wake_detected, command_after_wake = self.detect_wake_word(user_text)
        if wake_detected:
            if command_after_wake:
                return True, command_after_wake
            return True, "__interrupt_only__"

        return False, ""

    def should_process_after_wake(self, user_text: str, source: str = "voice"):
        if source == "typed" and not self.cfg.typed_input_requires_wake_word:
            return True, user_text

        if not self.cfg.require_wake_word:
            return True, user_text

        if self.wake_armed:
            self.wake_armed = False
            return True, user_text

        wake_detected, command_after_wake = self.detect_wake_word(user_text)
        if not wake_detected:
            print("[wake word not detected - ignored]")
            return False, ""

        if command_after_wake:
            return True, command_after_wake

        self.wake_armed = True
        self.tts_q.put("Yes?")
        print("[wake word detected - armed for next command]")
        return False, ""

    # ------------------------------------------------------------
    # Interruptible TTS
    # ------------------------------------------------------------

    def clear_tts_queue(self):
        try:
            while True:
                self.tts_q.get_nowait()
                self.tts_q.task_done()
        except queue.Empty:
            pass

    def interrupt_tts(self, reason: str = "interrupt"):
        if self.is_speaking:
            print(f"\n[interrupting speech: {reason}]")
            self.interrupt_event.set()
            self.clear_tts_queue()
            try:
                sd.stop()
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
                print(f"\nTTS error: {e}")
            finally:
                self.tts_q.task_done()

    # def speak_with_piper_interruptible(self, text: str):
    #     self.is_speaking = True
    #     self.interrupt_event.clear()

    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    #         wav_path = tmp.name

    #     try:
    #         if not os.path.exists(self.cfg.piper_model):
    #             print(f"Piper model not found: {self.cfg.piper_model}")
    #             return

    #         cmd = [
    #             self.cfg.piper_exe,
    #             "--model",
    #             self.cfg.piper_model,
    #             "--output_file",
    #             wav_path,
    #         ]

    #         try:
    #             result = subprocess.run(
    #                 cmd,
    #                 input=text.encode("utf-8"),
    #                 stdout=subprocess.PIPE,
    #                 stderr=subprocess.PIPE,
    #                 timeout=60,
    #             )
    #         except subprocess.TimeoutExpired:
    #             print("Piper timed out.")
    #             return

    #         if result.returncode != 0:
    #             print("Piper error:")
    #             print(result.stderr.decode(errors="ignore"))
    #             return

    #         if self.interrupt_event.is_set():
    #             return

    #         data, sr = sf.read(wav_path, dtype="float32")
    #         total = len(data) if data.ndim == 1 else data.shape[0]
    #         pos = 0
    #         chunk_size = max(256, int(self.cfg.playback_chunk_samples))

    #         # Chunked playback makes interruption responsive.
    #         while pos < total and not self.interrupt_event.is_set() and self.running:
    #             end = min(pos + chunk_size, total)
    #             sd.play(data[pos:end], sr)
    #             sd.wait()
    #             pos = end

    #         if self.interrupt_event.is_set():
    #             try:
    #                 sd.stop()
    #             except Exception:
    #                 pass

    #     finally:
    #         self.is_speaking = False
    #         self.interrupt_event.clear()
    #         try:
    #             os.remove(wav_path)
    #         except OSError:
    #             pass


    def speak_with_piper_interruptible(self, text: str):
        print(f"\n[TTS] Speaking: {text[:100]}")

        self.is_speaking = True
        self.interrupt_event.clear()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        print(f"[TTS] Temp file: {wav_path}")

    # ------------------------------------------------------------
    # Whisper
    # ------------------------------------------------------------

    def transcribe_pcm_bytes(self, pcm_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        try:
            self.save_wav_bytes(pcm_bytes, wav_path)
            segments, _ = self.whisper.transcribe(
                wav_path,
                beam_size=1,
                vad_filter=False,
                language="en",
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    # ------------------------------------------------------------
    # Ollama tools
    # ------------------------------------------------------------

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

    def ollama_chat(self, payload: dict) -> dict:
        response = requests.post(self.cfg.ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()

    def ollama_with_tools(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})

        first_payload = {
            "model": self.cfg.ollama_model,
            "messages": self.messages,
            "tools": TOOLS,
            "stream": False,
            "options": {"temperature": self.cfg.ollama_temperature},
        }

        try:
            first_data = self.ollama_chat(first_payload)
        except Exception as e:
            return f"Sorry, I could not contact Ollama. Error: {e}"

        message = first_data.get("message", {})
        self.messages.append(message)

        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            assistant_text = message.get("content", "").strip()
            return assistant_text or "I did not get a response."

        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name")
            args = self.parse_tool_args(function.get("arguments", {}))

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

        final_payload = {
            "model": self.cfg.ollama_model,
            "messages": self.messages,
            "stream": False,
            "options": {"temperature": self.cfg.ollama_temperature},
        }

        try:
            final_data = self.ollama_chat(final_payload)
        except Exception as e:
            return f"Tool completed, but final answer failed. Error: {e}"

        final_message = final_data.get("message", {})
        final_text = final_message.get("content", "").strip()

        if final_text:
            self.messages.append({"role": "assistant", "content": final_text})

        if len(self.messages) > 24:
            self.messages = [self.messages[0]] + self.messages[-22:]

        return final_text or "I found context, but I could not produce a final answer."

    # ------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------

    def process_user_text(self, user_text: str, source: str = "voice") -> bool:
        if not user_text:
            return True

        print(f"You ({source}): {user_text}")

        if self.is_exit_command(user_text):
            print("Exiting.")
            return False

        # Typed input is intentional, so interrupt immediately if TTS is active.
        if source == "typed" and self.is_speaking:
            self.interrupt_tts("typed command")

        # If voice arrives during TTS, accept only wake-word/interrupt phrase.
        if source == "voice" and self.is_speaking and self.cfg.smart_full_duplex:
            accepted, command_text = self.should_accept_voice_during_tts(user_text)
            if not accepted:
                print("[ignored during TTS: likely echo/noise/no wake word]")
                return True

            self.interrupt_tts("voice command during TTS")

            if command_text == "__interrupt_only__":
                print("[TTS stopped]")
                return True

            print(f"[command] {command_text}")
            answer = self.ollama_with_tools(command_text)
            print(f"AI: {answer}")
            self.tts_q.put(answer)
            return True

        should_process, command_text = self.should_process_after_wake(user_text, source=source)
        if not should_process:
            return True

        if self.is_exit_command(command_text):
            print("Exiting.")
            return False

        print(f"[command] {command_text}")
        self.is_processing = True
        try:
            answer = self.ollama_with_tools(command_text)
        finally:
            self.is_processing = False

        print(f"AI: {answer}")
        self.tts_q.put(answer)
        return True

    def handle_pending_typed_input(self) -> bool:
        processed_any = False
        while True:
            try:
                typed_text = self.text_q.get_nowait()
            except queue.Empty:
                break

            processed_any = True
            should_continue = self.process_user_text(typed_text, source="typed")
            self.text_q.task_done()

            if not should_continue:
                self.running = False
                return True

        return processed_any

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------

    def run(self):
        print()
        print("===================================================")
        print(" Local Realtime Voice Assistant v3.3 Smart Full-Duplex")
        print(" Mic -> Silero VAD -> Whisper -> Wake Word -> Ollama Tools/RAG -> Piper")
        print(" Features: typed input + smart voice interrupt + echo-resistant full duplex")
        print(" Voice examples:")
        print("   Hey assistant what is 45 times 23")
        print("   Hey assistant stop")
        print("   stop")
        print(" Keyboard: type a command and press Enter. Typed input interrupts TTS.")
        print(" Say/type: quit / exit / stop app / goodbye to quit.")
        print(" Note: for best ChatGPT-like duplex, use headphones.")
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

                    audio_tensor = self.bytes_to_float_tensor(chunk)
                    speech_event = self.vad_iterator(audio_tensor, return_seconds=False)

                    if speech_event:
                        if "start" in speech_event:
                            in_speech = True
                            speech_started_at = time.time()
                            speech_buffer = bytearray()
                            if self.is_speaking:
                                print("\n[start speech during TTS]")
                            else:
                                print("\n[start speech]")

                        if in_speech:
                            speech_buffer.extend(chunk)

                        if "end" in speech_event:
                            if self.is_speaking:
                                print("[end speech during TTS]")
                            else:
                                print("[end speech]")
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

                            if not user_text:
                                print("[empty transcription]")
                                self.vad_iterator.reset_states()
                                continue

                            should_continue = self.process_user_text(user_text, source="voice")
                            self.vad_iterator.reset_states()

                            if not should_continue:
                                self.running = False
                                break

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
                                    should_continue = self.process_user_text(user_text, source="voice")
                                    if not should_continue:
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


if __name__ == "__main__":
    config = AppConfig(
        ollama_model="qwen2.5:3b",
        piper_model="models/en_US-lessac-medium.onnx",
        piper_exe="piper",
        whisper_model="base",
        whisper_compute_type="int8",
        whisper_device="cpu",

        # Keep wake word on for speaker setups.
        require_wake_word=True,

        # Full-duplex smart mode.
        smart_full_duplex=True,
        allow_voice_interrupt_during_tts=True,

        # Keyboard input.
        enable_text_input=True,
        typed_input_requires_wake_word=False,
    )

    app = RealtimeVoiceAssistantV33(config)
    app.run()
