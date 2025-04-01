import pytest
from dataprocessor import DataProcessor

@pytest.fixture(scope="module")
def data_processor():
    return DataProcessor()

def test_very_postive(data_processor):
    text = "I love this song!"
    label, score = data_processor.sentiment_analysis(text)
    assert label in ["VERY POSITIVE", "POSITIVE"]
    assert score > 0.6

def test_very_negative(data_processor):
    text = "This is the worst thing ever. I'm so disappointed."
    label, score = data_processor.sentiment_analysis(text)
    assert label in ["VERY NEGATIVE", "NEGATIVE"]
    assert score > 0.6

def test_empty_returns_neutral(data_processor):
    text = ""
    label, score = data_processor.sentiment_analysis(text)
    assert label == 'NEUTRAL'
    assert score == 0.0

