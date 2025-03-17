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
2. **`transcriptionanalyser.py`** → Reads the generated CSV and creates **visualizations**.

---

## **🛠️ Setup Instructions**

### **1️⃣ Create a Virtual Environment**
```bash
python -m venv .venv
