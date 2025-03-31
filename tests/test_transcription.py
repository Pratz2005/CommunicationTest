import unittest
from dataprocessor import DataProcessor
import os

class TestTranscription(unittest.TestCase):
    def setUp(self):
        self.dp = DataProcessor()
        self.audio_path = "tests/test_audio.wav" 

    def test_transcription_output(self):
        expected_keywords = ['you', 'drove', 'through', 'me']
        self.assertTrue(os.path.exists(self.audio_path), f"File not found: {self.audio_path}")
        transcription = self.dp.transcribe_audio(self.audio_path)
        print("Transcription result:", transcription)
        
        for word in expected_keywords:
            self.assertIn(word, transcription.lower())

        self.assertGreater(len(transcription.strip()), 0)

if __name__ == '__main__':
    unittest.main()