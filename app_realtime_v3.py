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


@dataclass
class AppConfig:
    sample_rate: int = 16000
    channels: int = 1
    vad_window_samples: int = 512
    vad_threshold: float = 0.50
    min_silence_duration_ms: int = 600
    speech_pad_ms: int = 120
    #min_utterance_ms: int = 500 # barge-in increase
    min_utterance_ms: int = 1200 # barge-in increase
    max_utterance_seconds: int = 60 #20

    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"

    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_model: str = "qwen2.5:3b"

    piper_exe: str = "piper"
    piper_model: str = "models/en_US-lessac-medium.onnx"

    allow_barge_in: bool = False
    require_wake_word: bool = True

    wake_words: tuple = (
        "hey assistant",
        "hi assistant",
        "okay assistant",
        "ok assistant",
        "hello assistant",
        "hello ai"
    )


    system_prompt: str = (
        "You are a concise local voice assistant with tool access. "
        "Use rag_search for local documents, PDFs, requirements, architecture notes, guides, and project files. "
        "Use calculator for math. "
        "Use web_search for current internet information. "
        "When answering from rag_search, base the answer only on retrieved context and briefly mention sources. "
        "If retrieved context is insufficient, say so clearly. "
        "Keep answers short and natural for voice."
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
#         url = "https://duckduckgo.com/html/"
#         response = requests.get(
#             url,
#             params={"q": query},
#             timeout=15,
#             headers={"User-Agent": "Mozilla/5.0"},
#         )
#         if response.status_code != 200:
#             return f"Web search failed with status code {response.status_code}"
#         text = response.text
#         # Keep response compact for local LLM context.
#         return text[:3500]
#     except Exception as e:
#         return f"Web search error: {e}"


def web_search(query: str) -> str:
    try:
        url = "https://duckduckgo.com/html/"
        response = requests.get(
            url,
            params={"q": query},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if response.status_code != 200:
            return f"Web search failed with status code {response.status_code}"

        html = response.text

        # Very simple cleanup without extra dependencies.
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
                    "expression": {"type": "string", "description": "Math expression, for example: 45 * 23"}
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
                "properties": {"query": {"type": "string", "description": "Internet search query"}},
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
                "properties": {"query": {"type": "string", "description": "Semantic local document query"}},
            },
        },
    },
]


class RealtimeVoiceAssistantV3:
    def __init__(self, config: AppConfig):
        self.cfg = config
        self.audio_q = queue.Queue()
        self.tts_q = queue.Queue()
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

        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"\nAudio status: {status}")
        self.audio_q.put(bytes(indata))

    def bytes_to_float_tensor(self, audio_bytes: bytes) -> torch.Tensor:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(audio_np)

    def save_wav_bytes(self, pcm_bytes: bytes, path: str):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.cfg.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.sample_rate)
            wf.writeframes(pcm_bytes)

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

        # def detect_wake_word(self, text: str):
        #     normalized = self.normalize_text(text)
        #     for wake in self.cfg.wake_words:
        #         wake_norm = self.normalize_text(wake)
        #         if normalized == wake_norm:
        #             return True, ""
        #         if normalized.startswith(wake_norm + " "):
        #             return True, normalized[len(wake_norm):].strip()
        #         if wake_norm in normalized:
        #             before, after = normalized.split(wake_norm, 1)
        #             return True, after.strip()
        #     return False, ""

    # def detect_wake_word(self, text: str):
    #     original = text.strip()
    #     normalized = self.normalize_text(original)

    #     for wake in self.cfg.wake_words:
    #         wake_norm = self.normalize_text(wake)

    #         if normalized == wake_norm:
    #             return True, ""

    #         if normalized.startswith(wake_norm + " "):
    #             # Remove wake phrase from original text approximately.
    #             pattern = re.compile(re.escape(wake), re.IGNORECASE)
    #             cleaned = pattern.sub("", original, count=1).strip(" ,.!?")
    #             return True, cleaned

    #         if wake_norm in normalized:
    #             # Fallback: use normalized split if exact original removal is hard.
    #             _, after = normalized.split(wake_norm, 1)
    #             return True, after.strip()

    #     return False, ""
    
        def detect_wake_word(self, text: str):    def detect_wake_word = text.strip()
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


    def should_process_after_wake(self, user_text: str):
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

    def clear_tts_queue(self):
        try:
            while True:
                self.tts_q.get_nowait()
                self.tts_q.task_done()
        except queue.Empty:
            pass

    def interrupt_tts(self):
        #pass
        if self.is_speaking and self.cfg.allow_barge_in:
            print("\n[interrupting speech]")
            self.interrupt_event.set()
            self.clear_tts_queue()
            sd.stop()

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

    def ollama_with_tools(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        first_payload = {
            "model": self.cfg.ollama_model,
            "messages": self.messages,
            "tools": TOOLS,
            "stream": False,
            "options": {"temperature": 0.3},
        }
        try:
            first_response = requests.post(self.cfg.ollama_url, json=first_payload, timeout=120)
            first_response.raise_for_status()
            first_data = first_response.json()
        except Exception as e:
            return f"Sorry, I could not contact Ollama. Error: {e}"

        message = first_data.get("message", {})
        self.messages.append(message)
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            return message.get("content", "").strip() or "I did not get a response."

        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name")
            #args = function.get("arguments", {}) or {}
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
            #self.messages.append({"role": "tool", "tool_name": name, "content": str(result)})
            #self.messages.append({"role": "tool", "content": str(result)})
            
            self.messages.append({
                "role": "tool",
                "name": name,
                "content": str(result)
            })


        final_payload = {
            "model": self.cfg.ollama_model,
            "messages": self.messages,
            "stream": False,
            "options": {"temperature": 0.3},
        }
        try:
            final_response = requests.post(self.cfg.ollama_url, json=final_payload, timeout=120)
            final_response.raise_for_status()
            final_data = final_response.json()
        except Exception as e:
            return f"Tool completed, but final answer failed. Error: {e}"

        final_message = final_data.get("message", {})
        final_text = final_message.get("content", "").strip()
        if final_text:
            self.messages.append({"role": "assistant", "content": final_text})
        if len(self.messages) > 20:
            self.messages = [self.messages[0]] + self.messages[-18:]
        return final_text or "I found context, but I could not produce a final answer."

    def tts_worker(self):
        while self.running:
            try:
                text = self.tts_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if not text.strip():
                self.tts_q.task_done()
                continue
            if self.interrupt_event.is_set():
                self.tts_q.task_done()
                continue
            try:
                self.speak_with_piper(text)
            except Exception as e:
                print(f"\nTTS error: {e}")
            finally:
                self.tts_q.task_done()

    def speak_with_piper(self, text: str):
        self.is_speaking = True
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name
        try:
            if not os.path.exists(self.cfg.piper_model):
                print(f"Piper model not found: {self.cfg.piper_model}")
                return
            cmd = [self.cfg.piper_exe, "--model", self.cfg.piper_model, "--output_file", wav_path]


            try:
                result = subprocess.run(
                    cmd,
                    input=text.encode("utf-8"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                print("Piper timed out.")
                return


            if result.returncode != 0:
                print("Piper error:")
                print(result.stderr.decode(errors="ignore"))
                return
            if self.interrupt_event.is_set():
                return
            data, sr = sf.read(wav_path, dtype="float32")
            sd.play(data, sr)
            sd.wait()
        finally:
            self.is_speaking = False
            self.interrupt_event.clear()
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def run(self):
        print()
        print("===================================================")
        print(" Local Realtime Voice Assistant v3")
        print(" Mic → Silero VAD → Whisper → Wake Word → Ollama Tools/RAG → Piper")
        if self.cfg.require_wake_word:
            print(" Say 'Hey assistant or Hello AI' followed by your command. Press Ctrl+C to quit.")
        else:
            print(" Say something. Press Ctrl+C to quit.")
        print("===================================================")
        print()

        speech_buffer = bytearray()
        in_speech = False
        speech_started_at = None
        expected_bytes = self.cfg.vad_window_samples * 2

        try:
            with sd.RawInputStream(
                samplerate=self.cfg.sample_rate,
                blocksize=self.cfg.vad_window_samples,
                dtype="int16",
                channels=self.cfg.channels,
                callback=self.audio_callback,
            ):
                while self.running:
                    chunk = self.audio_q.get()
                    
                    # Prevent assistant from hearing its own speaker output.
                    # If barge-in is disabled, ignore microphone while TTS is playing.
                    if self.is_speaking and not self.cfg.allow_barge_in:
                        self.vad_iterator.reset_states()
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
                            if self.cfg.allow_barge_in:
                                self.interrupt_tts()
                            print("\n[start speech]")
                        if in_speech:
                            speech_buffer.extend(chunk)
                        if "end" in speech_event:
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
                            print(f"You: {user_text}")
                            normalized_text = self.normalize_text(user_text)
                            if normalized_text in {"quit", "exit", "stop", "goodbye"}:
                                print("Exiting.")
                                self.running = False
                                break
                            should_process, command_text = self.should_process_after_wake(user_text)
                            if not should_process:
                                self.vad_iterator.reset_states()
                                continue
                            print(f"[command] {command_text}")
                            answer = self.ollama_with_tools(command_text)
                            print(f"AI: {answer}")
                            self.tts_q.put(answer)
                            self.vad_iterator.reset_states()
                    else:
                        if in_speech:
                            speech_buffer.extend(chunk)
                            if speech_started_at and time.time() - speech_started_at > self.cfg.max_utterance_seconds:
                                print("[max utterance reached]")
                                in_speech = False
                                pcm = bytes(speech_buffer)
                                speech_buffer = bytearray()
                                user_text = self.transcribe_pcm_bytes(pcm)
                                if user_text:
                                    print(f"You: {user_text}")
                                    should_process, command_text = self.should_process_after_wake(user_text)
                                    if should_process:
                                        print(f"[command] {command_text}")
                                        answer = self.ollama_with_tools(command_text)
                                        print(f"AI: {answer}")
                                        self.tts_q.put(answer)
                                self.vad_iterator.reset_states()
        except KeyboardInterrupt:
            print("\nCtrl+C received. Exiting.")
        finally:
            self.running = False
            sd.stop()


# if __name__ == "__main__":
#     config = AppConfig(
#         ollama_model="qwen2.5:3b",
#         piper_model="models/en_US-lessac-medium.onnx",
#         piper_exe="piper",
#         whisper_model="base",
#         allow_barge_in=True,
#         require_wake_word=True,
#     )
#     app = RealtimeVoiceAssistantV3(config)
#     app.run()


if __name__ == "__main__":
    config = AppConfig(
        ollama_model="qwen2.5:3b",
        piper_model="models/en_US-lessac-medium.onnx",
        piper_exe="piper",
        whisper_model="base",
        allow_barge_in=False,
        require_wake_word=True,
    )
    app = RealtimeVoiceAssistantV3(config)
    app.run()
