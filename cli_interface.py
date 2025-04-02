from dataprocessor import DataProcessor
from transcriptionanalyzer import TranscriptionAnalyzer
from chunking_strategy import FixedLengthChunking, SilenceBasedChunking

class CLIInterface:
    def __init__(self):
        self.bucket_name = "mycarvideobucket"
        self.output_folder = "output_csv"

    def run(self):
        print("Welcome to the Video Transcription Pipeline!")
        while True:
            print("\nWhat would you like to do?")
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
        print("\nSelect chunking strategy:")
        print("1. Fixed-length (every 5 seconds)")
        print("2. Silence-based (split when there's a pause)")
        strategy_choice = input("Enter your choice (1/2): ").strip()

        if strategy_choice == "1":
            chunker = FixedLengthChunking(segment_length=5)
        elif strategy_choice == "2":
            chunker = SilenceBasedChunking()
        else:
            print("Invalid strategy selected. Defaulting to Fixed-Length.")
            chunker = FixedLengthChunking(segment_length=5)

        processor = DataProcessor(output_folder=self.output_folder)
        processor.chunker = chunker  
        print(f"\nProcessing using: {chunker.__class__.__name__}\n")
        processor.process_all_videos(bucket_name=self.bucket_name)
        print("Video processing completed!\n")

    def analyze_data(self):
        print(f"\nLoading CSVs from {self.output_folder}...\n")
        analyzer = TranscriptionAnalyzer(self.output_folder)
        analyzer.load_data()
        analyzer.plot_word_count_histogram()
        analyzer.plot_sentiment_distribution()
        analyzer.compare_chunking_strategies()
        print("Analysis completed!\n")

if __name__ == "__main__":
    interface = CLIInterface()
    interface.run()
