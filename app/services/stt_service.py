import asyncio
import io
import wave

import numpy as np
import webrtcvad

from app.schemas import TranscriptionResponse, SegmentResult, PostProcessedSegment
from app.services.post_processor import PostProcessor


class STTService:
    def __init__(self, model, executor, config):
        self.model = model
        self.executor = executor
        self.config = config
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        self.post_processor = PostProcessor(config)

    async def transcribe(self, audio_bytes):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._run, audio_bytes)

    def _run(self, audio_bytes):
        audio = self._decode(audio_bytes)
        raw = self._transcribe(self._segment(audio))
        post = self._re_segment(raw)
        for seg in post:
            seg.text = self.post_processor.process(seg.text)
        return TranscriptionResponse(raw_results=raw, post_processed_results=post)

    def _decode(self, data):
        with wave.open(io.BytesIO(data), "rb") as wf:
            nchannels, sampwidth, framerate, nframes = (
                wf.getnchannels(),
                wf.getsampwidth(),
                wf.getframerate(),
                wf.getnframes(),
            )
            raw = wf.readframes(nframes)

        if nframes == 0:
            raise ValueError("empty WAV")

        if sampwidth == 1:
            audio = np.frombuffer(raw, np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sampwidth == 2:
            audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 3:
            tmp = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
            tmp = tmp[:, 0] | (tmp[:, 1] << 8) | (tmp[:, 2] << 16)
            mask = 1 << 23
            audio = ((tmp ^ mask) - mask).astype(np.float32) / 8388608.0
        elif sampwidth == 4:
            audio = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported sample width: {sampwidth}")

        if nchannels > 1:
            audio = audio.reshape(-1, nchannels).mean(axis=1)

        if framerate != self.config.sample_rate:
            n = int(len(audio) * self.config.sample_rate / framerate)
            x_old = np.linspace(0, len(audio), len(audio), endpoint=False)
            x_new = np.linspace(0, len(audio), n, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)

        return audio

    def _segment(self, audio):
        sr = self.config.sample_rate
        frame_size = int(sr * self.config.frame_duration_ms / 1000)

        int16 = (audio * 32768.0).astype(np.int16)
        frames = []
        for i in range(0, len(int16), frame_size):
            f = int16[i : i + frame_size]
            if len(f) < frame_size:
                f = np.pad(f, (0, frame_size - len(f)), "constant")
            frames.append(f)

        speech = []
        for f in frames:
            try:
                speech.append(self.vad.is_speech(f.tobytes(), sr))
            except Exception:
                speech.append(False)

        silence_frames = int(self.config.silence_threshold_s * 1000 / self.config.frame_duration_ms)
        segs = []
        start = None
        sil_count = 0

        for i, s in enumerate(speech):
            if s:
                if start is None:
                    start = i
                sil_count = 0
            elif start is not None:
                sil_count += 1
                if sil_count >= silence_frames:
                    end_sample = min((i - silence_frames + 1) * frame_size, len(audio))
                    if end_sample > start * frame_size:
                        segs.append((start * frame_size / sr, audio[start * frame_size : end_sample]))
                    start = None
                    sil_count = 0

        if start is not None and start * frame_size < len(audio):
            segs.append((start * frame_size / sr, audio[start * frame_size :]))

        return segs or [(0.0, audio)]

    def _transcribe(self, segments):
        results = []
        for offset, seg_audio in segments:
            segs, _ = self.model.transcribe(seg_audio, beam_size=5, vad_filter=False, language="en")
            for s in segs:
                text = s.text.strip()
                if text:
                    results.append(
                        SegmentResult(
                            start=round(offset + s.start, 3),
                            end=round(offset + s.end, 3),
                            text=text,
                            avg_logprob=round(float(s.avg_logprob), 4),
                        )
                    )
        return results

    def _re_segment(self, segments):
        if not segments:
            return []

        threshold = self.config.confidence_threshold
        merged = []

        for seg in segments:
            low = seg.avg_logprob < threshold
            if merged and merged[-1].low_confidence and low:
                last = merged[-1]
                last.text = f"{last.text} {seg.text}".strip()
                last.end = seg.end
                last.avg_logprob = round((last.avg_logprob + seg.avg_logprob) / 2, 4)
            else:
                merged.append(
                    PostProcessedSegment(
                        start=seg.start,
                        end=seg.end,
                        text=f"[Low Confidence] {seg.text}" if low else seg.text,
                        avg_logprob=seg.avg_logprob,
                        low_confidence=low,
                    )
                )

        return merged
