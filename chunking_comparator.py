import os
import pandas as pd
import matplotlib.pyplot as plt

class ChunkingComparator:
    def __init__(self, folder="output_csv"):
        self.folder = folder
        self.paired_files = self._pair_csvs()

    def _pair_csvs(self):
        files = [f for f in os.listdir(self.folder) if f.endswith('.csv')]
        video_map = {}

        for file in files:
            base = file.rsplit('_', 1)[0]  # removes _fixedlength / _silencebased
            video_map.setdefault(base, []).append(file)

        return {k: v for k, v in video_map.items() if len(v) == 2}

    def compare_csvs(self, file1, file2):
        df1 = pd.read_csv(os.path.join(self.folder, file1))
        df2 = pd.read_csv(os.path.join(self.folder, file2))

        return {
            'chunks': (len(df1), len(df2)),
            'avg_words': (
                df1['Transcription'].fillna('').apply(lambda x: len(x.split())).mean(),
                df2['Transcription'].fillna('').apply(lambda x: len(x.split())).mean()
            ),
            'avg_confidence': (df1['Confidence'].mean(), df2['Confidence'].mean())
        }

    def plot_comparison(self):
        for video, files in self.paired_files.items():
            file1, file2 = sorted(files)  # sort to get consistent order
            result = self.compare_csvs(file1, file2)

            labels = ['FixedLength', 'SilenceBased']

            # Plot chunks
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.bar(labels, result['chunks'], color=['skyblue', 'salmon'])
            plt.title('Number of Segments')

            # Plot average word count
            plt.subplot(1, 3, 2)
            plt.bar(labels, result['avg_words'], color=['skyblue', 'salmon'])
            plt.title('Avg Words per Segment')

            # Plot confidence
            plt.subplot(1, 3, 3)
            plt.bar(labels, result['avg_confidence'], color=['skyblue', 'salmon'])
            plt.title('Avg Sentiment Confidence')

            plt.suptitle(f"Comparison for: {video}")
            plt.tight_layout()
            plt.show()

    def summary_report(self):
        for video, files in self.paired_files.items():
            file1, file2 = sorted(files)
            result = self.compare_csvs(file1, file2)

            print(f"\nVideo: {video}")
            print(f"  - Segments:       Fixed: {result['chunks'][0]}, Silence: {result['chunks'][1]}")
            print(f"  - Avg Words:      Fixed: {result['avg_words'][0]:.2f}, Silence: {result['avg_words'][1]:.2f}")
            print(f"  - Avg Confidence: Fixed: {result['avg_confidence'][0]:.2f}, Silence: {result['avg_confidence'][1]:.2f}")
