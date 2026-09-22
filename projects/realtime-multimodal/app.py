"""Gradio voice/text client for the citation-safe streaming API."""
import io
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import gradio as gr
import httpx
import numpy as np


@lru_cache(maxsize=1)
def whisper():
    if os.getenv("ENABLE_LOCAL_STT", "0") != "1":
        raise ValueError("Local speech recognition is disabled. Type your question or enable ENABLE_LOCAL_STT.")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ValueError("Install requirements-voice.txt to enable local speech recognition.") from exc
    return WhisperModel(os.getenv("STT_MODEL", "tiny.en"), device="cpu", compute_type="int8")


def speech(text):
    if not shutil.which("espeak"):
        return None
    result = subprocess.run(["espeak", "--stdout", "--stdin"], input=text.encode(),
                            capture_output=True, timeout=10, check=True)
    with wave.open(io.BytesIO(result.stdout), "rb") as audio:
        return (audio.getframerate(), np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16))


def respond(question, audio, read_aloud):
    start = time.perf_counter()
    marks = {}
    try:
        if audio:
            with wave.open(audio, "rb") as recording:
                seconds = recording.getnframes() / recording.getframerate()
            if seconds > 30:
                raise ValueError("Keep recordings under 30 seconds.")
            yield "Transcribing locally…", "", None
            segments, _ = whisper().transcribe(audio, beam_size=1)
            question = " ".join(segment.text for segment in segments).strip()
            marks["transcription_ms"] = round((time.perf_counter() - start) * 1000)
        if not 3 <= len(question.strip()) <= 2000:
            raise ValueError("Enter a question between 3 and 2,000 characters.")
        url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        key = os.getenv("API_KEY", "")
        yield "Retrieving evidence…", json.dumps(marks), None
        with httpx.stream("POST", url + "/stream", json={"question": question},
                          headers={"Authorization": f"Bearer {key}"} if key else {},
                          timeout=httpx.Timeout(30, connect=3)) as response:
            response.raise_for_status()
            received = False
            request_start = time.perf_counter()
            for line in response.iter_lines():
                if time.perf_counter() - request_start > 30:
                    raise TimeoutError("Request deadline exceeded")
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "error":
                    raise ValueError(event["message"])
                if event["type"] == "stage":
                    marks[event["stage"] + "_ms"] = round((time.perf_counter() - start) * 1000)
                    yield "Working: " + event["stage"], json.dumps(marks, indent=2), None
                if event["type"] == "answer":
                    received = True
                    marks["answer_ms"] = round((time.perf_counter() - start) * 1000)
                    answer = event["answer"]
                    for citation in event["citations"] or event["retrieved"]:
                        answer += "\n\n" + citation["source"] + ": " + citation.get("quote", citation.get("text", ""))
                    details = {"trace_id": event["trace_id"], "status": event["status"], **marks}
                    yield answer, json.dumps(details, indent=2), None
                    if read_aloud and event["status"] == "ok" and event["citations"]:
                        try:
                            output = speech(event["answer"])
                            details["tts_ready_ms"] = round((time.perf_counter() - start) * 1000)
                            if output is None:
                                details["tts_note"] = "Install espeak for offline speech output; text remains available."
                            yield answer, json.dumps(details, indent=2), output
                        except (OSError, subprocess.SubprocessError, wave.Error):
                            details["tts_note"] = "Speech output failed; validated text remains available."
                            yield answer, json.dumps(details, indent=2), None
            if not received:
                raise ValueError("Incomplete response; please retry")
    except ValueError as exc:
        yield "Unable to complete: " + str(exc), json.dumps(marks), None
    except (httpx.HTTPError, TimeoutError, OSError, wave.Error):
        yield "Backend or audio unavailable. Check deployment configuration and try typed input.", json.dumps(marks), None


def build_app():
    with gr.Blocks(title="Real-Time Multimodal Application", delete_cache=(3600, 3600)) as demo:
        gr.Markdown("# Real-Time Multimodal Application\n"
                    "Citation-safe voice and text interaction · AI Portfolio Projects\n\n"
                    "Local STT is optional; microphone recordings go to this app server. "
                    "Record at most 30 seconds. Text works without speech model downloads.")
        question = gr.Textbox(label="Question", placeholder="When must API keys be rotated?", max_lines=4)
        audio = gr.Audio(sources=["microphone"], type="filepath", format="wav", label="Optional voice question")
        read_aloud = gr.Checkbox(label="Read validated answer aloud using offline eSpeak", value=False)
        with gr.Row():
            submit = gr.Button("Ask my docs", variant="primary")
            stop = gr.Button("Stop waiting")
        answer = gr.Textbox(label="Answer and evidence", lines=10)
        timings = gr.Textbox(label="Measured stages (milliseconds from submission)")
        output_audio = gr.Audio(label="Validated answer audio")
        event = submit.click(respond, [question, audio, read_aloud], [answer, timings, output_audio])
        stop.click(fn=None, cancels=[event])
        gr.Markdown("Stopping cancels queued/generator work, not a native STT kernel or an in-flight blocking "
                    "HTTP read immediately. Worker isolation is needed for hard compute cancellation.")
    return demo.queue(max_size=8, default_concurrency_limit=2)


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")),
                       max_file_size="5mb", show_error=False)
