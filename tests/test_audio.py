import os
import pytest
from dataprocessor import DataProcessor
from chunking_strategy import FixedLengthChunking, SilenceBasedChunking

@pytest.fixture(scope="module")
def data_processor():
    return DataProcessor()

@pytest.fixture(scope="module")
def test_audio_path():
    path = "tests/test_audio.wav"
    assert os.path.exists(path), f"File not found: {path}"
    return path

def test_fixed_length_chunking(test_audio_path):
    dp = DataProcessor()
    dp.chunker = FixedLengthChunking(segment_length=2)  # Use fixed-length strategy

    segments = dp.segment_audio(test_audio_path)
    assert len(segments) >= 1
    for start, end, file_path in segments:
        assert os.path.exists(file_path)
        assert file_path.endswith(".wav")
        assert end - start > 0

def test_silence_based_chunking(test_audio_path):
    dp = DataProcessor()
    dp.chunker = SilenceBasedChunking(min_silence_len=500, silence_thresh=-40)  # Use silence strategy

    segments = dp.segment_audio(test_audio_path)
    assert len(segments) >= 1
    for _, _, file_path in segments:
        assert os.path.exists(file_path)
        assert file_path.endswith(".wav")
