import os
import torch
import pandas as pd
import librosa
from moviepy import *
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import botocore
import tempfile
import shutil

class DataProcessor:
    def __init__(self, output_folder="output_csv", model_name="openai/whisper-small"):
        if os.path.exists("videos"):
            shutil.rmtree("videos")

        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        # Load Whisper Model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)

        # Load Sentiment Analysis Model
        self.classifier = pipeline("sentiment-analysis", model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    def list_s3_videos(self, bucket_name):
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        response = s3.list_objects_v2(Bucket=bucket_name)
        return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.mp4')]

    def download_video_from_s3(self, bucket_name, s3_key):
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        video_name = os.path.splitext(os.path.basename(s3_key))[0]
        
        #Create a folder of the video name
        video_folder = os.path.join("videos",video_name)
        os.makedirs(video_folder, exist_ok=True)

        local_path = os.path.join(video_folder, os.path.basename(s3_key))

        with open(local_path, "wb") as f:
            s3.download_fileobj(bucket_name, s3_key, f)

        print(f"Downloaded {s3_key} to {local_path}")
        return local_path
    
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
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_folder = os.path.dirname(video_path)


        audio_path = os.path.join(video_folder, f"{video_name}.wav")
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

    def process_all_videos(self, bucket_name):
        s3_keys = self.list_s3_videos(bucket_name)

        for s3_key in s3_keys:
            local_path = self.download_video_from_s3(bucket_name, s3_key)
            self.process_video(local_path)

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all_videos(bucket_name="mycarvideobucket")