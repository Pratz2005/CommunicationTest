# Video Transcription & Sentiment Analysis Pipeline

This project processes **video files** by:
1. **Extracting audio** from the video.
2. **Splitting the audio into 5-second segments**.
3. **Transcribing speech** using OpenAI's **Whisper** model.
4. **Performing sentiment analysis** on each transcribed segment using sentiment-analysis on HuggingFace.
5. **Saving the processed data to a CSV file**.
6. **Generating a histogram of word counts per 5-second interval**.
7. **Creating a sentiment distribution bar chart**.

This project consists of two Python scripts:
1. **`dataprocessor.py`** → Processes video files and generates a CSV with transcriptions and sentiments.
2. **`transcriptionanalyzer.py`** → Reads the generated CSV and creates **visualizations**.

---
##  **Requirements**
1. moviepy
2. transformers
3. speechrecognition
4. pandas
5. matplotlib
6. librosa
7. torch

## **Setup Instructions**

## **Create a directory named dataset_videos and add the relevant videos in it**

Since the driving video was too large it could not be pushed to github so I have not added it

1.Clone the Repo
```bash
git clone https://github.com/Pratz2005/CommunicationTest.git 
cd communicationtest
```

2.**Make sure u add the relevant videos on which you want to run this pipeline to the dataset_videos directory**

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

### Now you can run the dataprocessor.py file and then the transcriptionanalyzer.py file and see the results.

## **Testing & Execution**
This section describes how the programs were executed and tested.

### **Testing `dataprocessor.py`**
- Used multiple `.mp4` video files in `dataset_videos/` to ensure **batch processing** works.
- Verified that each video’s **audio was extracted correctly**.
- Confirmed that **Whisper transcribed the speech** without errors.
- Ensured **sentiment analysis correctly categorized** transcriptions as **POSITIVE, NEGATIVE, or NEUTRAL**.
- Checked that **CSV files were generated correctly** in `output_csv/`.

### **Testing `transcriptionanalyser.py`**
- Loaded different **CSV files** to ensure proper **data handling**.
- Verified that the **histogram displayed word count correctly**.
- Confirmed that the **sentiment bar chart displayed appropriate classifications**.

### **Edge Cases Considered**
- **Empty videos / silent parts** → Checked that **no empty transcriptions caused errors**.
- **Long videos** → Ensured **large files didn't cause memory issues**.
- **High-noise environments** → Verified that **Whisper handled speech well in noisy audio**.
- **Multiple sentiment patterns** → Ensured **diverse phrases were classified accurately**.




