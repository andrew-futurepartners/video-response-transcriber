import os
import warnings
import whisper
import openai
import pandas as pd
import json
import re

# 1) suppress Whisper’s CPU warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# 2) ensure ffmpeg is on PATH (if needed)
os.environ["PATH"] += os.pathsep + r"C:\Users\apier\PycharmProjects\Express Explore Transcriber\ffmpeg\ffmpeg\bin"

# 3) your OpenAI key
openai.api_key = "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"

# 4) map each question_label → full question text
QUESTION_TEXT_MAP = {
    "qEEProud": "What makes you most proud about living in Little Rock?",
    "qEETourism": "When thinking about visitors to Little Rock, what is one positive thing that comes to mind?",
    "qEEImprovement": "What is one change or improvement that would make Little Rock better for BOTH visitors and residents?"
}

# 5) define roots
VIDEOS_ROOT = "videos"
DATA_FILES_ROOT = "data files"

# 6) load Whisper model (base for testing)
print("🧠 Loading Whisper 'large' model…")
model = whisper.load_model("large")

results = []

# 7) iterate through each question folder
for folder in os.listdir(VIDEOS_ROOT):
    folder_path = os.path.join(VIDEOS_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    parts = folder.split("_")
    if len(parts) < 5:
        print(f"🔶 Skipping unexpected folder name: {folder}")
        continue

    survey_id      = parts[3]
    question_label = parts[4]
    question_text  = QUESTION_TEXT_MAP.get(question_label)
    if not question_text:
        print(f"🔶 No question-text mapping for label '{question_label}', skipping.")
        continue

    # read media_list.txt
    list_path = os.path.join(folder_path, "media_list.txt")
    if not os.path.exists(list_path):
        print(f"🔶 No media_list.txt in {folder}, skipping.")
        continue

    with open(list_path, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f if line.strip()]

    for filename in filenames:
        base_name = os.path.splitext(filename)[0]
        video_path = os.path.join(folder_path, "Total", filename)
        if not os.path.exists(video_path):
            print(f"⚠️ Missing video file: {video_path}, skipping.")
            continue

        print(f"\n🎬 [{survey_id} – {question_label}] Transcribing: {filename}")
        # a) transcribe with Whisper
        try:
            wout = model.transcribe(video_path)
            transcript = wout["text"].strip()
        except RuntimeError:
            transcript = "[NO AUDIO]"
            score      = 0
            flag       = "yes"
            results.append({
                "survey_id": survey_id,
                f"{question_label}_File": base_name,
                question_label: transcript,
                f"{question_label}_Score": score,
                f"{question_label}_Flag": flag
            })
            continue

        # b) score with GPT-4 (no redaction)
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
        print("🤖 Sending to GPT-4 for scoring…")
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a transcription reviewer."},
                    {"role": "user",   "content": prompt.strip()}
                ],
                temperature=0.2,
            )
            content = resp["choices"][0]["message"]["content"]
            match = re.search(r"\{.*?\}", content, re.DOTALL)
            if not match:
                raise ValueError("No valid JSON in GPT response.")
            parsed = json.loads(match.group(0))

            score            = parsed.get("score", 0)
            clean_transcript = parsed.get("clean_transcript", "").strip() or "[NO AUDIO]"
            flag             = "yes" if score < 50 else "no"

        except Exception as e:
            print(f"❌ GPT error on {filename}: {e}")
            clean_transcript = "[GPT ERROR]"
            score            = 0
            flag             = "yes"

        results.append({
            "survey_id": survey_id,
            f"{question_label}_File": base_name,
            question_label: clean_transcript,
            f"{question_label}_Score": score,
            f"{question_label}_Flag": flag
        })

# 8) compile into flat results DataFrame
results_df = pd.DataFrame(results)
merged_dfs = []

# 9) for each survey, merge into its data file and collect trimmed DataFrames
for survey_id in results_df["survey_id"].unique():
    data_file = os.path.join(DATA_FILES_ROOT, f"{survey_id}.xlsx")
    if not os.path.exists(data_file):
        print(f"⚠️ No data file for survey {survey_id}, skipping merge.")
        continue
    print(f"🔁 Merging into data file: {os.path.basename(data_file)}")

    base_df = pd.read_excel(data_file)

    sub_df  = results_df[results_df["survey_id"] == survey_id]

    # perform per-question mapping on base_df
    for qlabel in QUESTION_TEXT_MAP.keys():
        file_col = f"{qlabel}_File"
        if file_col not in sub_df.columns or qlabel not in base_df.columns:
            continue

        # rename base key column to <question_label>_File
        base_df.rename(columns={qlabel: file_col}, inplace=True)

        trans_map = sub_df.set_index(file_col)[qlabel].to_dict()
        score_map = sub_df.set_index(file_col)[f"{qlabel}_Score"].to_dict()
        flag_map  = sub_df.set_index(file_col)[f"{qlabel}_Flag"].to_dict()

        base_df[qlabel]              = base_df[file_col].map(trans_map).fillna("")
        base_df[f"{qlabel}_Score"] = base_df[file_col].map(score_map).fillna(0).astype(int)
        base_df[f"{qlabel}_Flag"]  = base_df[file_col].map(flag_map).fillna("no")

    # add survey_id to base_df
    base_df['survey_id'] = survey_id

    # trim to only video response related columns & respondents
    file_cols = [f"{qlabel}_File" for qlabel in QUESTION_TEXT_MAP.keys() if f"{qlabel}_File" in base_df.columns]
    q_cols = []
    for qlabel in QUESTION_TEXT_MAP.keys():
        if qlabel in base_df.columns:
            q_cols.extend([qlabel, f"{qlabel}_Score", f"{qlabel}_Flag"])

    cols_to_keep = ['survey_id'] + file_cols + q_cols
    trimmed_df = base_df[cols_to_keep]
    mask = trimmed_df[file_cols].apply(lambda row: any(val and str(val).strip() for val in row), axis=1)
    trimmed_df = trimmed_df[mask]

    # save new transcribed data file (without overwriting original)
    transcribed_file = os.path.join(DATA_FILES_ROOT, f"{survey_id}_Transcribed.xlsx")
    try:
        base_df.to_excel(transcribed_file, index=False)
        print(f"✅ Saved Transcribed File: {os.path.basename(transcribed_file)}")
    except PermissionError:
        alt = os.path.join(DATA_FILES_ROOT, f"{survey_id}_Transcribed_copy.xlsx")
        base_df.to_excel(alt, index=False)
        print(f"⚠️ File locked—saved to {os.path.basename(alt)}")

    merged_dfs.append(trimmed_df)

# 10) write global flat results
output_raw = "transcription_results.xlsx"
try:
    results_df.to_excel(output_raw, index=False)
    print(f"\n✅ Global results saved to → {output_raw}")
except PermissionError:
    alt = "transcription_results_copy.xlsx"
    results_df.to_excel(alt, index=False)
    print(f"⚠️ Global file locked—saved to → {alt}")

# 11) if any merged DataFrames exist, concat and write respondent-level file
if merged_dfs:
    combined_df = pd.concat(merged_dfs, ignore_index=True)
    output_resp = "transcription_results_respondents.xlsx"
    try:
        combined_df.to_excel(output_resp, index=False)
        print(f"\n✅ Respondent-level results saved to → {output_resp}")
    except PermissionError:
        alt = "transcription_results_respondents_copy.xlsx"
        combined_df.to_excel(alt, index=False)
        print(f"⚠️ Respondent file locked—saved to → {alt}")
else:
    print("⚠️ No data files found; skipped respondent-level merge.")