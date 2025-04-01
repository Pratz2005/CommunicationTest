import os
import pytest
from dataprocessor import DataProcessor

@pytest.fixture(scope="module")
def data_processor():
    return DataProcessor()

@pytest.fixture(scope="module")
def audio_path():
    path = "tests/test_audio.wav"
    assert os.path.exists(path), f"File not found: {path}"
    return path

def test_transcription_output_contains_expected_words(data_processor, audio_path):
    expected_keywords = ['you', 'drove', 'through', 'me']
    transcription = data_processor.transcribe_audio(audio_path)
    print("Transcription:", transcription)

    assert isinstance(transcription, str)
    assert transcription.strip() != ""

    for word in expected_keywords:
        assert word in transcription.lower(), f"Missing expected word: {word}"
