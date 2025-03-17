import os
import torch
import pandas as pd
import librosa
from moviepy import *
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

class DataProcessor:
    def __init__(self, dataset_folder="dataset_videos", output_folder="output_csv", model_name="openai/whisper-small"):
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        # Load Whisper Model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)

        # Load Sentiment Analysis Model
        self.classifier = pipeline("sentiment-analysis")

    def extract_audio(self, video_path, audio_path):
        """
        Extracts audio from a video file and saves it as a WAV file.
        """
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)  # Ensure 16kHz format

    def segment_audio(self, audio_path, segment_length=5):
        """
        Segments audio into smaller chunks of `segment_length` seconds.
        """
        audio = AudioFileClip(audio_path)
        duration = int(audio.duration)
        segments = []

        for start in range(0, duration, segment_length):
            end = min(start + segment_length, duration)
            segment_path = f"{audio_path[:-4]}_{start}-{end}.wav"
            audio.subclipped(start, end).write_audiofile(segment_path, codec='pcm_s16le')
            segments.append((start, end, segment_path))

        return segments

    def transcribe_audio(self, segment_path):
        # Load and preprocess the audio
        speech_array, _ = librosa.load(segment_path, sr=16000)
        input_features = self.processor(
            speech_array, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)

        # Force English transcription to avoid automatic translation
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="en", task="transcribe"
        )

        # Generate transcription with attention mask
        predicted_ids = self.model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            attention_mask=torch.ones_like(input_features)
        )

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()

    def sentiment_analysis(self, text):
        """
        Performs sentiment analysis on the transcribed text.
        """
        if not text.strip():
            return 'NEUTRAL'

        sentiment = self.classifier(text)[0]
        return sentiment['label']

    def process_video(self, video_path):
        """
        Processes a single video file: extracts audio, segments it, transcribes, analyzes sentiment, and saves results to CSV.
        """
        audio_path = video_path.replace('.mp4', '.wav')
        self.extract_audio(video_path, audio_path)

        segments = self.segment_audio(audio_path)
        data = []

        for start, end, segment_path in segments:
            transcription = self.transcribe_audio(segment_path)
            sentiment = self.sentiment_analysis(transcription)
            data.append([start, end, transcription, sentiment])

        # Save to CSV
        df = pd.DataFrame(data, columns=['Start_Time', 'End_Time', 'Transcription', 'Sentiment'])
        output_csv = os.path.join(self.output_folder, os.path.basename(video_path).replace(".mp4", ".csv"))
        df.to_csv(output_csv, index=False)
        print(f"Processed: {output_csv}")

    def process_all_videos(self):
        """
        Iterates through all videos in the dataset folder and processes them.
        """
        for file in os.listdir(self.dataset_folder):
            if file.endswith(".mp4"):
                self.process_video(os.path.join(self.dataset_folder, file))

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all_videos()
