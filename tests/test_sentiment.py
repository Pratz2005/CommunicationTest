import unittest
from dataprocessor import DataProcessor

class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        self.dp = DataProcessor()

    def test_very_positive(self):
        text = "I love this song!"
        label, score = self.dp.sentiment_analysis(text)
        self.assertIn(label, ['VERY POSITIVE', 'POSITIVE'])
        self.assertGreater(score, 0.6)

    def test_very_negative(self):
        text = "This is the worst thing ever. I'm so disappointed."
        label, score = self.dp.sentiment_analysis(text)
        self.assertIn(label, ['VERY NEGATIVE', 'NEGATIVE'])
        self.assertGreater(score, 0.6)

    def test_empty_text(self):
        label, score = self.dp.sentiment_analysis("")
        self.assertEqual(label, 'NEUTRAL')

if __name__ == '__main__':
    unittest.main()
