import os
import warnings
import whisper
import pandas as pd
import zipfile
from dotenv import load_dotenv
from pathlib import Path
from visual_checks_lite import VisualInspectorLite
import time, random, openai, json
from openai.error import RateLimitError, APIConnectionError, Timeout

vis_inspector = VisualInspectorLite(
    max_frames=12,
    dark_thresh=0.5,     # feel free to revert to 0.3
    face_threshold=0.3,
    blur_thresh=150.0,
)

def gpt_json_call(messages,
                  model="gpt-4o-mini",
                  max_retries=3,
                  **kwargs) -> dict:
    """
    Robust JSON-mode ChatCompletion.
    Retries on rate-limit, network glitches, or timeouts.
    Returns the parsed JSON dict.
    """
    for attempt in range(max_retries):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
                timeout=30,          # seconds
                **kwargs,
            )
            return json.loads(resp.choices[0].message.content)

        except (RateLimitError, APIConnectionError, Timeout) as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"🔄 GPT retry {attempt+1}/{max_retries} after {e} "
                  f"(sleep {wait:.1f}s)")
            time.sleep(wait)

        except Exception:
            # unknown error → don't loop endlessly
            raise
    # all retries exhausted
    raise RuntimeError("GPT call failed after retries")

def has_any_video(row) -> bool:
    """
    Return True iff at least one *_File column contains a non-empty string.
    NaN, None, 0, etc. are treated as “no video”.
    """
    for v in row:
        if isinstance(v, str) and v.strip():      # ← only real filenames pass
            return True
    return False

# 1) suppress Whisper’s CPU warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# 2) ensure ffmpeg is on PATH (if needed)
os.environ["PATH"] += os.pathsep + r"C:\Users\apier\PycharmProjects\Express Explore Transcriber\ffmpeg\ffmpeg\bin"

# 3) your OpenAI key
# === GLOBAL SETTINGS ===
load_dotenv()  # take environment variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

# 4) map each question_label → full question text
QUESTION_TEXT_MAP = {
    "qEEPerceptions1": "What three words would you use to describe your time in Atlantic City?",
    "qEEPerceptions2": "What would motivate you to return to Atlantic City for a leisure trip within the next year?",
    "qEEPerceptions3": "What would have convinced you to extend your most recent trip to Atlantic City for at least one or more nights?"
}

# 5) define roots
VIDEOS_ROOT = "videos"
DATA_FILES_ROOT = "data files"

# 6) load Whisper model medium
print("🧠 Loading Whisper 'medium' model…")
model = whisper.load_model("base")

# 6.5) unzip all .zip files inside VIDEOS_ROOT
for item in os.listdir(VIDEOS_ROOT):
    if item.lower().endswith(".zip"):
        zip_path = os.path.join(VIDEOS_ROOT, item)
        extract_dir = os.path.join(VIDEOS_ROOT, os.path.splitext(item)[0])

        if os.path.isdir(extract_dir):
            print(f"📂 Skipping already extracted: {item}")
            continue

        print(f"🗜️ Extracting {item} → {extract_dir}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"✅ Unzipped: {item}")
        except zipfile.BadZipFile:
            print(f"❌ Failed to unzip {item} (BadZipFile)")

# 6.6) unzip any .zip archives inside DATA_FILES_ROOT ------------------------
for item in os.listdir(DATA_FILES_ROOT):
    if not item.lower().endswith(".zip"):
        continue

    zip_path = os.path.join(DATA_FILES_ROOT, item)
    # if an .xlsx with the same survey_id is already there, skip
    survey_id = os.path.splitext(item)[0]          # 250403.zip → 250403
    xlsx_path = os.path.join(DATA_FILES_ROOT, f"{survey_id}.xlsx")
    if os.path.exists(xlsx_path):
        print(f"📂 Data zip already extracted: {item}")
        continue

    print(f"🗜️  Extracting data zip {item} → {DATA_FILES_ROOT}")
    try:
        # ⬇️  SINGLE LINE that flattens the archive into the root folder
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_FILES_ROOT)
        print(f"✅  Unzipped data file: {item}")
    except zipfile.BadZipFile:
        print(f"❌  Corrupt data zip: {item}")

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
            wout = model.transcribe(video_path, language="en", task="transcribe")
            transcript = wout["text"].strip()
            transcript = wout["text"].strip()

            # ---------- NEW: handle silent / empty transcripts ----------
            if not transcript or len(transcript.split()) < 3:
                transcript = "[NO AUDIO]"
                score = 0
                flag = "yes"
                vis_score = vis_inspector.analyse_video(Path(video_path))["score"]

                results.append({
                    "survey_id": survey_id,
                    f"{question_label}_File": base_name,
                    question_label: transcript,
                    f"{question_label}_Score": score,
                    f"{question_label}_Flag": flag,
                    f"{question_label}_VisualScore": vis_score,
                })
                continue  # skip GPT scoring for this clip
        except RuntimeError:
            transcript = "[NO AUDIO]"
            score      = 0
            flag       = "yes"
            visual_score = vis_inspector.analyse_video(Path(video_path))["score"]
            results.append({
                "survey_id": survey_id,
                f"{question_label}_File": base_name,
                question_label: transcript,
                f"{question_label}_Score": score,
                f"{question_label}_Flag": flag,
                f"{question_label}_VisualScore": visual_score
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
        print("🤖  Scoring with GPT-4…")
        try:
            parsed = gpt_json_call(
                [
                    {"role": "system", "content": "Respond in JSON only."},
                    {"role": "user", "content": prompt.strip()},
                ]
            )
            score = int(parsed.get("score", 0))
            clean_transcript = parsed.get("clean_transcript", "").strip() or "[NO AUDIO]"
            flag = "yes" if score < 50 else "no"

        except Exception as e:
            print(f"❌ GPT error on {filename}: {e}")
            clean_transcript = "[GPT ERROR]"
            score = 0
            flag = "yes"

        vis_score = vis_inspector.analyse_video(Path(video_path))["score"]

        results.append(
            {
                "survey_id": survey_id,
                f"{question_label}_File": base_name,
                question_label: clean_transcript,
                f"{question_label}_Score": score,
                f"{question_label}_Flag": flag,
                f"{question_label}_VisualScore": vis_score,
            }
        )

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

        # rename only if it hasn’t been renamed before
        if qlabel in base_df.columns and file_col not in base_df.columns:
            base_df.rename(columns={qlabel: file_col}, inplace=True)

        if file_col not in sub_df.columns or file_col not in base_df.columns:
            continue

        trans_map = sub_df.set_index(file_col)[qlabel].to_dict()
        score_map = sub_df.set_index(file_col)[f"{qlabel}_Score"].to_dict()
        flag_map  = sub_df.set_index(file_col)[f"{qlabel}_Flag"].to_dict()
        if f"{qlabel}_VisualScore" in sub_df.columns:
            visual_map = sub_df.set_index(file_col)[f"{qlabel}_VisualScore"].to_dict()
            base_df[f"{qlabel}_VisualScore"] = base_df[file_col].map(visual_map).fillna(0).astype(int)

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
            q_cols.extend([
                qlabel,
                f"{qlabel}_Score",
                f"{qlabel}_Flag",
                f"{qlabel}_VisualScore"
            ])

    cols_to_keep = ['survey_id'] + file_cols + q_cols
    trimmed_df = base_df[cols_to_keep]
    mask = trimmed_df[file_cols].apply(has_any_video, axis=1)
    trimmed_df = trimmed_df[mask].copy()

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
    # Compute final composite score across content and visual scores
    score_cols = [col for col in combined_df.columns if col.endswith("_Score") or col.endswith("_VisualScore")]
    combined_df["Final_Quality_Score"] = combined_df[score_cols].mean(axis=1).round(1)

    # ✅ Re-trim after merge to remove any remaining empty rows
    video_cols = [col for col in combined_df.columns if col.endswith("_File")]
    combined_df = combined_df[
        combined_df[video_cols].apply(lambda row: any(val and str(val).strip() for val in row), axis=1)
    ]

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