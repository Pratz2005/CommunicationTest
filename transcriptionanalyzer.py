import pandas as pd
import matplotlib.pyplot as plt
import os
from chunking_comparator import ChunkingComparator

class TranscriptionAnalyzer:
    def __init__(self, folder_path):
        self.folder_path = folder_path

        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder '{self.folder_path}' does not exist. Please run transcription first.")
        
        self.data_files = self.get_csv_files()

        if not self.data_files:
            raise FileNotFoundError(f"No CSV files found in folder '{self.folder_path}'. Did you run transcription?")

        self.data_frames = {}
        self.comparator = ChunkingComparator(folder=folder_path)

    def get_csv_files(self):
        return [f for f in os.listdir(self.folder_path) if f.endswith(".csv")]

    def load_data(self):
        if not self.data_files:
            raise FileNotFoundError(f"No CSV files found in {self.folder_path}")

        for file_name in self.data_files:
            file_path = os.path.join(self.folder_path, file_name)
            df = pd.read_csv(file_path)

            # Convert timestamps to numeric values
            df["Start_Time"] = pd.to_numeric(df["Start_Time"], errors='coerce')
            df["End_Time"] = pd.to_numeric(df["End_Time"], errors='coerce')

            # Count words in each transcription segment
            df["Word_Count"] = df["Transcription"].fillna("").apply(lambda x: len(x.split()))
            df["Sentiment"] = df["Sentiment"].replace({
                "SLIGHTLY POSITIVE": "S. POS",
                "SLIGHTLY NEGATIVE": "S. NEG",
                "VERY POSITIVE": "V. POS",
                "VERY NEGATIVE": "V. NEG"
            })

            self.data_frames[file_name] = df

    def plot_word_count_histogram(self):
        if not self.data_frames:
            raise ValueError("No data loaded. Call load_data() first.")
        
        for file_name, df in self.data_frames.items():
            if df['Start_Time'].isnull().all():
                continue
            plt.figure(figsize=(12, 6))
            plt.bar(df["Start_Time"], df["Word_Count"], width=4.5, align='edge', color='skyblue', edgecolor='black')
            plt.xlabel("Time (seconds)")
            plt.ylabel("Word Count")
            plt.title(f"Histogram of Transcribed Words per 5-Second Interval\n({file_name})")
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

    def plot_word_count_by_segment(self):
        if not self.data_frames:
            raise ValueError("No data loaded. Call load_data() first.")
        
        for file_name, df in self.data_frames.items():
            if df["Start_Time"].isnull().all():  # If silence-based, no time info
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(df)), df["Word_Count"], color='purple', edgecolor='black')
                plt.xlabel("Segment Index")
                plt.ylabel("Word Count")
                plt.title(f"Histogram of Words per Silence-Based Segment\n({file_name})")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.show()

    def plot_sentiment_distribution(self):
        if not self.data_frames:
            raise ValueError("No data loaded. Call load_data() first.")

        color_map = {"NEUTRAL": "gray", "POSITIVE": "green", "NEGATIVE": "red", "V. POS": "lime", "V. NEG": "darkred", "S. NEG": "lightcoral", "S. POS": "darkgreen"}

        for file_name, df in self.data_frames.items():
            sentiment_counts = df["Sentiment"].value_counts()

            colors = [color_map[sent] for sent in sentiment_counts.index]

            plt.figure(figsize=(8, 6))
            sentiment_counts.plot(kind='bar', color=colors, edgecolor='black')
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            plt.title(f"Distribution of Sentiments in Transcription\n({file_name})")
            plt.xticks(rotation=0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

    def compare_chunking_strategies(self):
        print("\nCHUNKING STRATEGY COMPARISON")
        self.comparator.summary_report()
        self.comparator.plot_comparison()
