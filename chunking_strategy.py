from pydub import AudioSegment, silence
import os
from moviepy import *

class ChunkingStrategy:
    def segment(self, audio_path):
        raise NotImplementedError("Subclasses must implement this!")

class FixedLengthChunking(ChunkingStrategy):
    def __init__(self, segment_length=5):
        self.segment_length = segment_length

    def segment(self, audio_path):
        audio = AudioFileClip(audio_path)
        duration = int(audio.duration)
        segments = []

        for start in range(0, duration, self.segment_length):
            end = min(start + self.segment_length, duration)
            segment_path = f"{audio_path[:-4]}_{start}-{end}.wav"
            audio.subclipped(start, end).write_audiofile(segment_path, codec='pcm_s16le')
            segments.append((start, end, segment_path))

        return segments

class SilenceBasedChunking(ChunkingStrategy):
    def __init__(self, min_silence_len=500, silence_thresh=-40):
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh

    def segment(self, audio_path):
        sound = AudioSegment.from_wav(audio_path)
        chunks = silence.split_on_silence(
            sound, 
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
            keep_silence=100
        )

        segments = []
        for idx, chunk in enumerate(chunks):
            segment_path = f"{audio_path[:-4]}_chunk{idx}.wav"
            chunk.export(segment_path, format="wav")
            segments.append((None, None, segment_path))

        return segments
