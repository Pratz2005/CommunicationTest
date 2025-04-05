# Video Transcription & Sentiment Analysis Pipeline

This project processes **video files** by:
1. **Extracting the video** from s3 buckets
2. **Extracting audio** from the video.
2. **Splitting the audio into 5-second segments/silence-based chunking**.
3. **Transcribing speech** using OpenAI's **Whisper** model.
4. **Performing sentiment analysis** on each transcribed segment using sentiment-analysis on HuggingFace.
5. **Saving the processed data to a CSV file** - 2 files- one for fixed segment and one for silence-based chunking.
6. **Generating a histogram of word counts per 5-second interval**.
7. **Creating a sentiment distribution bar chart**.
8. **Comparing the two chunking strategies**.

This project consists of two Python scripts:
1. **`dataprocessor.py`** → Processes video files and generates a CSV with transcriptions and sentiments.
2. **`transcriptionanalyser.py`** → Reads the generated CSV and creates **visualizations**.
3. **`cli_interface.py`**  → Is the Command Line Interface for the User.
4. **`chunking_strategy.py`** → Has classes that implement fixed length chunking and silence based chunking
5. **`chunking_comparator.py`** → Does the comparison for the two chunking strategies

Note: You can generate csvs for both chunking strategies just by chaninging two lines of code n the data_processor.py file - importing the chunker and initialising it in the __init__ function. How to do so is given in the comments in the code.
---


## **Setup Instructions**

1.Clone the Repo
```bash
git clone https://github.com/Pratz2005/CommunicationTest.git 
cd communicationtest
```

### **Create a Virtual Environment**
```bash
python -m venv .venv
```

### **Activate the .venv**
```bash
.venv\Scripts\activate
source .venv/bin/activate //For MacOS/Linux
```
### **install requirements.txt**
```bash
pip install -r requirements.txt
```

### Now you can run the cli_interface.py file and see the results. This file has CLI user interface and calls the dataprocessor and transcriptionanalyzer class to return the chunked audio files and a csv containing the sentiment analysis and confidence of each chunk within each chunking strategy file

### Tests
### I also implemented basic unit tests for testing the audio, transcription and sentiment using pytests

```bash
cd communicationtest
pytest tests
```

### I also compared the output of the various chunking strategies in the transcriptionanalyzer.py file







