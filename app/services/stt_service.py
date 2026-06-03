import asyncio
import io
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
import webrtcvad
from faster_whisper import WhisperModel

from app.config import Settings
from app.schemas import TranscriptionResponse, SegmentResult, PostProcessedSegment


class STTService:
    def __init__(self, model: WhisperModel, executor: ThreadPoolExecutor, config: Settings):
        self.model = model
        self.executor = executor
        self.config = config
        self._vad = webrtcvad.Vad(config.vad_aggressiveness)

    async def transcribe(self, audio_bytes: bytes) -> TranscriptionResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._pipeline,
            audio_bytes,
        )

    def _pipeline(self, audio_bytes: bytes) -> TranscriptionResponse:
        audio = self._decode_wav(audio_bytes)
        segments = self._vad_segment(audio)
        raw_results = self._transcribe_segments(segments)
        post_processed = self._confidence_merge(raw_results)
        return TranscriptionResponse(
            raw_results=raw_results,
            post_processed_results=post_processed,
        )

    def _decode_wav(self, audio_bytes: bytes) -> np.ndarray:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if n_frames == 0:
            raise ValueError("WAV file contains no audio frames")

        if sampwidth == 1:
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 3:
            audio = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            audio = audio[:, 0] | (audio[:, 1] << 8) | (audio[:, 2] << 16)
            mask = 1 << 23
            audio = (audio ^ mask) - mask
            audio = audio.astype(np.float32) / 8388608.0
        elif sampwidth == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        if framerate != self.config.sample_rate:
            audio = self._resample(audio, framerate, self.config.sample_rate)

        return audio

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        n_samples = int(len(audio) * target_sr / orig_sr)
        x_old = np.linspace(0, len(audio), len(audio), endpoint=False)
        x_new = np.linspace(0, len(audio), n_samples, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    def _vad_segment(self, audio: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        frame_size = int(self.config.sample_rate * self.config.frame_duration_ms / 1000)

        int_audio = (audio * 32768.0).astype(np.int16)

        frames = []
        for i in range(0, len(int_audio), frame_size):
            frame = int_audio[i : i + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)), "constant")
            frames.append(frame)

        is_speech = []
        for frame in frames:
            try:
                is_speech.append(self._vad.is_speech(frame.tobytes(), self.config.sample_rate))
            except Exception:
                is_speech.append(False)

        segments = []
        speech_start = None
        silence_count = 0
        silence_frames = int(
            self.config.silence_threshold_s * 1000 / self.config.frame_duration_ms
        )

        for i, speech in enumerate(is_speech):
            if speech:
                if speech_start is None:
                    speech_start = i
                silence_count = 0
            else:
                if speech_start is not None:
                    silence_count += 1
                    if silence_count >= silence_frames:
                        start_sample = speech_start * frame_size
                        end_sample = (i - silence_frames + 1) * frame_size
                        end_sample = min(end_sample, len(audio))
                        if end_sample > start_sample:
                            seg_audio = audio[start_sample:end_sample]
                            start_time = start_sample / self.config.sample_rate
                            segments.append((start_time, seg_audio))
                        speech_start = None
                        silence_count = 0

        if speech_start is not None:
            start_sample = speech_start * frame_size
            if start_sample < len(audio):
                seg_audio = audio[start_sample:]
                start_time = start_sample / self.config.sample_rate
                segments.append((start_time, seg_audio))

        if not segments:
            segments.append((0.0, audio))

        return segments

    def _transcribe_segments(
        self, segments: List[Tuple[float, np.ndarray]]
    ) -> List[SegmentResult]:
        results = []
        for start_time, seg_audio in segments:
            segs, _ = self.model.transcribe(
                seg_audio,
                beam_size=5,
                vad_filter=False,
                language="en",
            )
            for s in segs:
                text = s.text.strip()
                if not text:
                    continue
                results.append(
                    SegmentResult(
                        start=round(start_time + s.start, 3),
                        end=round(start_time + s.end, 3),
                        text=text,
                        avg_logprob=round(float(s.avg_logprob), 4),
                    )
                )
        return results

    def _confidence_merge(
        self, segments: List[SegmentResult]
    ) -> List[PostProcessedSegment]:
        if not segments:
            return []

        threshold = self.config.confidence_threshold
        merged: List[PostProcessedSegment] = []

        for seg in segments:
            is_low = seg.avg_logprob < threshold
            if merged and merged[-1].low_confidence and is_low:
                last = merged[-1]
                last.text = f"{last.text} {seg.text}".strip()
                last.end = seg.end
                last.avg_logprob = round(
                    (last.avg_logprob + seg.avg_logprob) / 2, 4
                )
            else:
                label = f"[Low Confidence] {seg.text}" if is_low else seg.text
                merged.append(
                    PostProcessedSegment(
                        start=seg.start,
                        end=seg.end,
                        text=label,
                        avg_logprob=seg.avg_logprob,
                        low_confidence=is_low,
                    )
                )

        return merged
