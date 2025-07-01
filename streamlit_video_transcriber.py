import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import whisper
import openai
import json
import re
from io import BytesIO

# === PAGE CONFIGURATION ===
st.set_page_config(page_title="Video Response Transcriber", layout="wide")
st.title("🎥 Survey Video Response Transcriber")

# === GLOBAL SETTINGS ===
from dotenv import load_dotenv
import os
load_dotenv()  # take environment variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = whisper.load_model("large")

# === SESSION STATE ===
if "question_texts" not in st.session_state:
    st.session_state.question_texts = {}

# === FILE UPLOADS ===
st.header("1. Upload Video Folders")
folders = st.file_uploader("Upload one or more zipped folders (each representing a question batch)", type="zip", accept_multiple_files=True)

data_files = st.file_uploader("(Optional) Upload original data file(s) for merging", type=["xlsx"], accept_multiple_files=True)

process_button = st.button("🔁 Process Videos")

# === PROCESSING ===
if process_button and folders:
    with st.spinner("Processing uploaded folders..."):
        temp_root = tempfile.mkdtemp()
        results = []
        datafile_map = {}

        # Save uploaded data files by survey_id
        for df_file in data_files or []:
            survey_match = re.search(r'(\w{4}_\d+)', df_file.name)
            if survey_match:
                survey_id = survey_match.group(1)
                datafile_map[survey_id] = df_file

        for zip_file in folders:
            folder_name = zip_file.name.replace(".zip", "")
            folder_path = os.path.join(temp_root, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(folder_path)

            parts = folder_name.split("_")
            if len(parts) < 5:
                st.warning(f"Skipping unexpected folder name format: {folder_name}")
                continue

            survey_id = parts[3]
            question_label = parts[4]

            # Prompt for question text
            if question_label not in st.session_state.question_texts:
                st.session_state.question_texts[question_label] = st.text_input(
                    f"Enter question text for {question_label}:", key=question_label)

            question_text = st.session_state.question_texts[question_label]

            media_list_path = os.path.join(folder_path, "media_list.txt")
            video_dir = os.path.join(folder_path, "Total")

            if not os.path.exists(media_list_path):
                st.warning(f"Missing media_list.txt in {folder_name}")
                continue

            with open(media_list_path, "r", encoding="utf-8") as f:
                filenames = [line.strip() for line in f if line.strip()]

            for filename in filenames:
                base_name = os.path.splitext(filename)[0]
                video_path = os.path.join(video_dir, filename)

                if not os.path.exists(video_path):
                    st.warning(f"Missing video file: {filename}")
                    continue

                try:
                    wout = MODEL.transcribe(video_path)
                    transcript = wout["text"].strip()
                except RuntimeError:
                    transcript = "[NO AUDIO]"
                    score = 0
                    flag = "yes"
                    results.append({
                        "survey_id": survey_id,
                        f"{question_label}_File": base_name,
                        question_label: transcript,
                        f"{question_label}_Score": score,
                        f"{question_label}_Flag": flag
                    })
                    continue

                prompt = f"""
You are an expert qualitative analyst reviewing video transcriptions.  
You must score each transcript from 0 to 100 based on how well it answers the question.  
Do not redact or alter the transcript in any way—return it unchanged.

Question: {question_text}

Transcript: {transcript}

Instructions:
1. Score the quality of the response from 0–100.
2. Do not change the transcript.
3. Return only a JSON object:
   {{
     "score": <0-100>,
     "clean_transcript": "<original transcript>"
   }}
"""
                try:
                    resp = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a transcription reviewer."},
                            {"role": "user", "content": prompt.strip()}
                        ],
                        temperature=0.2,
                    )
                    content = resp["choices"][0]["message"]["content"]
                    match = re.search(r"\{.*?\}", content, re.DOTALL)
                    parsed = json.loads(match.group(0))
                    score = parsed.get("score", 0)
                    clean_transcript = parsed.get("clean_transcript", "[NO AUDIO]").strip()
                    flag = "yes" if score < 50 else "no"
                except Exception as e:
                    clean_transcript = "[GPT ERROR]"
                    score = 0
                    flag = "yes"

                results.append({
                    "survey_id": survey_id,
                    f"{question_label}_File": base_name,
                    question_label: clean_transcript,
                    f"{question_label}_Score": score,
                    f"{question_label}_Flag": flag
                })

        if results:
            df = pd.DataFrame(results)
            st.subheader("🔎 Preview of Results")
            st.dataframe(df.head())

            # Download results
            st.download_button("📥 Download Results (Excel)", df.to_excel(index=False), file_name="transcription_results.xlsx")
        else:
            st.error("No results generated.")
