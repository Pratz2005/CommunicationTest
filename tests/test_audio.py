import unittest
import os
from dataprocessor import DataProcessor

class TestAudioSegmentation(unittest.TestCase):
    def setUp(self):
        self.dp = DataProcessor()

    def test_segmentation_output(self):
        segments = self.dp.segment_audio("tests/test_audio.wav", segment_length=2)

        # Ensure we got at least one segment
        self.assertGreaterEqual(len(segments), 1)

        # Check each segment
        for start, end, file in segments:
            self.assertTrue(os.path.exists(file), f"Segment file missing: {file}")
            self.assertTrue(file.endswith('.wav'), f"Incorrect format: {file}")
            self.assertGreater(end - start, 0, f"Invalid segment duration: {start}–{end}")

if __name__ == '__main__':
    unittest.main()