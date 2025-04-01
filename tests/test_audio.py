import os
import pytest
from dataprocessor import DataProcessor

@pytest.fixture(scope="module")
def data_processor():
    return DataProcessor()

@pytest.fixture(scope="module")
def test_audio_path():
    path = "tests/test_audio.wav"
    assert os.path.exists(path), f"File not found: {path}"
    return path

def test_audio_segmentation_output(data_processor, test_audio_path):
    segments = data_processor.segment_audio(test_audio_path, segment_length=2)

    assert len(segments) >= 1, "No segments were created."

    for start, end, file_path in segments:
        assert os.path.exists(file_path), f"Segment file missing: {file_path}"
        assert file_path.endswith(".wav"), f"Invalid segment format: {file_path}"
        assert end - start > 0, f"Invalid segment duration: {start}–{end}"
