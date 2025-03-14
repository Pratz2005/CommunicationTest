from moviepy import *
import os
from transformers import pipeline
import speech_recognition as sr
import pandas as pd

class DataProcessor:
    def __init__(self, dataset_folder="dataset_videos", output_folder="output_csv"):
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.classifier = pipeline("sentiment-analysis")

    def extract_audio(self,video_path, audio_path):
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path, codec='pcm_s16le')

    def segment_audio(self, audio_path, segment_length = 5):
        audio = AudioFileClip(audio_path)
        duration = int(audio.duration)
        segments = []

        for start in range(0,duration,segment_length):
            end = min(start + segment_length, duration)
            segment_path = f"{audio_path[:-4]}_{start}-{end}.wav"
            audio.subclipped(start,end).write_audiofile(segment_path, codec='pcm_s16le')
            segments.append((start,end,segment_path))

        return segments            


    def transcribe_audio(self, segment_path):
        recognizer = sr.Recognizer()
        with sr.AudioFile(segment_path) as source:
            audio = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return ""
            except sr.RequestError:
                return ""

    def sentiment_analysis(self, text):
        if not text.strip():
            return 'NEUTRAL'
        
        sentiment = self.classifier(text)[0]
        return sentiment['label']

    def process_video(self, video_path):
        audio_path = video_path.replace('mp4', 'wav')
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
        print(output_csv)

    def process_all_videos(self):
        for file in os.listdir(self.dataset_folder):
            if file.endswith(".mp4"):
                self.process_video(os.path.join(self.dataset_folder, file))

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all_videos()




