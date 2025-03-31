from dataprocessor import DataProcessor
from transcriptionanalyzer import TranscriptionAnalyzer

class CLIInterface:
    def __init__(self):
        self.bucket_name = "mycarvideobucket"
        self.output_folder = "output_csv"

    def run(self):
        print("Welcome to the Video Transcription Pipeline!")
        while True:
            print("What would you like to do?")
            print("1. Process videos from S3")
            print("2. Analyze transcriptions")
            print("3. Exit")

            choice = input("Enter your choice (1/2/3): ").strip()

            if choice == "1":
                self.process_videos()
            elif choice == "2":
                self.analyze_data()
            elif choice == "3":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Try again.")

    def process_videos(self):
        print(f"\nProcessing videos from bucket: {self.bucket_name}...\n")
        processor = DataProcessor(output_folder=self.output_folder)
        processor.process_all_videos(bucket_name=self.bucket_name)
        print("Video processing completed!\n")

    def analyze_data(self):
        print(f"\nLoading CSVs from {self.output_folder}...\n")
        analyzer = TranscriptionAnalyzer(self.output_folder)
        analyzer.load_data()
        analyzer.plot_word_count_histogram()
        analyzer.plot_sentiment_distribution()
        print("Analysis completed!\n")

if __name__ == "__main__":
    interface = CLIInterface()
    interface.run()