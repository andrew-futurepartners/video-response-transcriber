import os
import warnings
import whisper
import openai
import pandas as pd
import json
import re
import zipfile
import shutil
from dotenv import load_dotenv
from openai import OpenAI


# 0) load env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env")

# 1) suppress Whisper CPU warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# 2) ensure ffmpeg is on PATH (if needed)
os.environ["PATH"] += os.pathsep + r"C:\Users\apier\PycharmProjects\Express Explore Transcriber\ffmpeg\ffmpeg\bin"

# 3) unzip functionality
def unzip_folder(zip_path, extract_to):
    """Unzip a folder if it hasn't been unzipped already"""
    if not os.path.exists(zip_path):
        return False
    
    # Check if already unzipped
    base_name = os.path.splitext(os.path.basename(zip_path))[0]
    expected_extract_path = os.path.join(extract_to, base_name)
    
    if os.path.exists(expected_extract_path):
        print(f"📁 {base_name} already unzipped, skipping...")
        return True
    
    try:
        # Create the target directory for this specific zip file
        os.makedirs(expected_extract_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files in the zip to debug
            file_list = zip_ref.namelist()
            print(f"🔍 Files in {os.path.basename(zip_path)}: {len(file_list)} files")
            
            # Extract all files
            zip_ref.extractall(expected_extract_path)
            
            # Verify extraction by checking for Total folder
            total_path = os.path.join(expected_extract_path, "Total")
            if os.path.exists(total_path):
                total_files = os.listdir(total_path)
                print(f"✅ Extracted {len(total_files)} files to Total/ folder")
            else:
                print(f"⚠️ No Total folder found after extraction")
                # List what was actually extracted
                extracted_items = os.listdir(expected_extract_path)
                print(f"📂 Extracted items: {extracted_items}")
        
        print(f"📦 Unzipped {os.path.basename(zip_path)} to {base_name}/")
        return True
    except Exception as e:
        print(f"❌ Error unzipping {zip_path}: {e}")
        return False

def unzip_all_folders(root_dir):
    """Unzip all zip files in a directory"""
    if not os.path.exists(root_dir):
        print(f"⚠️ Directory {root_dir} does not exist")
        return
    
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if item.endswith('.zip') and os.path.isfile(item_path):
            unzip_folder(item_path, root_dir)

# 4) map each question_label -> full question text
QUESTION_TEXT_MAP = {
    "qOneThingEE": "What is the one thing Las Vegas has that sets it apart from other vacation destinations?",
    "qFriendEE": "If a friend asked you “is Las Vegas worth it?”, what would you say — and why? Please be specific and describe any activities, attractions, experiences or vibes you would mention to them, as well as any comparisons you might make to other destinations.",
    "qPosSurpriseEE": "What has been the most pleasantly surprising part of your Vegas trip so far? What will you tell your friends or family about it when you get home?",
    "qNegSurpriseEE": "What has been the most unfortunately frustrating part of your Vegas trip so far? What will you tell your friends or family about it when you get home?",
    "qFeesCurrentEE": "Have there been any costs or charges you hadn’t anticipated on this trip? How have you encountered them, how are they affecting your impression of Las Vegas overall, and how do they compare to other destinations you have recently visited?",
    "qMemoryEE": "What is the strongest memory you have about your most recent visit to Las Vegas, and how does this memory make you feel when you recall it?",
    "qFeesRecentEE": "On your most recent trip to Las Vegas, were there any costs or charges you hadn’t anticipated? How did you encounter them, how did they affect your impression of Las Vegas overall, and how do they compare to other destinations you have recently visited?",
    "qInspireEE": "Think about planning your next vacation – what might inspire you to return to Las Vegas? A fond memory or feeling, the people and atmosphere, a certain experience, a great deal, or something else?",
    "qMissingEE": "Is there something you feel is missing from the Las Vegas experience that has kept you from visiting yet? Please fill in the blank in your response: “If Las Vegas had _________, I would consider booking a trip right now.”"
}
# 5) helper functions for processing different submission types
def get_submission_type_from_folder(folder_name):
    """Extract submission type (_TXT, _AUD, _VID) from folder name"""
    if '_AUD_' in folder_name:
        return 'AUD'
    elif '_VID_' in folder_name:
        return 'VID'
    elif '_TXT_' in folder_name:
        return 'TXT'
    return None

def get_base_question_from_folder(folder_name):
    """Extract base question name from folder name (e.g., qOneThingEE from qOneThingEE_AUD)"""
    if '_AUD_' in folder_name:
        return folder_name.split('_AUD_')[0].split('_')[-1]  # Get the question part
    elif '_VID_' in folder_name:
        return folder_name.split('_VID_')[0].split('_')[-1]  # Get the question part
    elif '_TXT_' in folder_name:
        return folder_name.split('_TXT_')[0].split('_')[-1]  # Get the question part
    return folder_name

def transcribe_media_file(file_path, model):
    """Transcribe audio or video file using Whisper"""
    try:
        wout = model.transcribe(file_path)
        return wout["text"].strip()
    except RuntimeError as e:
        print(f"⚠️ Transcription error for {file_path}: {e}")
        return "[NO AUDIO]"

def score_transcript_with_gpt(transcript, question_text, client):
    """Score transcript using GPT API"""
    prompt = f"""
You are an expert qualitative analyst reviewing video transcriptions.
You must score each transcript from 0 to 100 based on how well it answers the question.
Do not redact or alter the transcript in any way, return it unchanged.

Question: {question_text}

Transcript: {transcript}

Instructions:
1. Score the quality of the response from 0 to 100.
2. Do not change the transcript.
3. Return only a JSON object:
   {{
     "score": <0-100>,
     "clean_transcript": "<original transcript>"
   }}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a transcription reviewer."},
                {"role": "user", "content": prompt.strip()},
            ],
            temperature=0.1,
            seed=42,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON in GPT response.")
        parsed = json.loads(match.group(0))
        
        score = parsed.get("score", 0)
        clean_transcript = parsed.get("clean_transcript", "").strip() or "[NO AUDIO]"
        flag = "yes" if score < 50 else "no"
        
        return clean_transcript, score, flag
    except Exception as e:
        print(f"❌ GPT error: {e}")
        return "[GPT ERROR]", 0, "yes"

# 6) define roots
VIDEOS_ROOT = "videos"
DATA_FILES_ROOT = "data files"

# Unzip all folders
print("📦 Checking for zip files to extract...")
unzip_all_folders(VIDEOS_ROOT)
unzip_all_folders(DATA_FILES_ROOT)

# 7) load Whisper model and initialize OpenAI client
print("🧠 Loading Whisper 'base' model...")
model = whisper.load_model("base")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 8) collect all submissions by question and survey
submissions_by_question = {}

# Process all folders in videos directory
for folder in os.listdir(VIDEOS_ROOT):
    folder_path = os.path.join(VIDEOS_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue
    
    # Extract submission type and base question
    submission_type = get_submission_type_from_folder(folder)
    if not submission_type:
        print(f"🔶 Skipping folder without submission type: {folder}")
        continue
    
    base_question = get_base_question_from_folder(folder)
    if base_question not in QUESTION_TEXT_MAP:
        print(f"🔶 No question mapping for '{base_question}', skipping.")
        continue
    
    # Extract survey_id from folder name
    # Format: selfserve.decipherinc.com_selfserve_38ba_250904_qFeesUpdatedEE_AUD_media_testimonials
    # The survey ID is the 4th element (index 3) when split by underscore
    parts = folder.split("_")
    if len(parts) < 5:
        print(f"🔶 Unexpected folder format: {folder}")
        continue
    
    survey_id = parts[3]  # Fourth element is the survey ID (e.g., 250904)
    
    print(f"   - Extracted survey_id: {survey_id}")
    
    # Initialize structure if needed
    if base_question not in submissions_by_question:
        submissions_by_question[base_question] = {}
    if survey_id not in submissions_by_question[base_question]:
        submissions_by_question[base_question][survey_id] = {'TXT': [], 'AUD': [], 'VID': []}
    
    # Process media files in this folder
    list_path = os.path.join(folder_path, "media_list.txt")
    if not os.path.exists(list_path):
        print(f"🔶 No media_list.txt in {folder}, skipping.")
        continue
    
    with open(list_path, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f if line.strip()]
    
    # Check if media_list.txt is empty
    if not filenames:
        print(f"⚠️ Empty media_list.txt in {folder}, no media files to process.")
        continue
    
    print(f"   - Media list contains {len(filenames)} files")
    
    # Check if Total folder exists
    total_path = os.path.join(folder_path, "Total")
    if not os.path.exists(total_path):
        print(f"⚠️ No Total folder in {folder}, checking for media files in root...")
        # Look for media files in the root of the folder
        media_files = []
        all_files = os.listdir(folder_path)
        print(f"   - Files in root directory: {all_files}")
        
        for file in all_files:
            if file.lower().endswith(('.mp4', '.mp3', '.wav', '.avi', '.mov', '.m4a')):
                media_files.append(file)
        
        if not media_files:
            print(f"❌ No media files found in {folder}, skipping.")
            print(f"   - This suggests the zip file only contained empty media_list.txt")
            continue
        else:
            print(f"📁 Found {len(media_files)} media files in root directory")
            filenames = media_files
            total_path = folder_path  # Use root directory instead of Total subfolder
    else:
        print(f"✅ Total folder found with {len(os.listdir(total_path))} files")
    
    for filename in filenames:
        base_name = os.path.splitext(filename)[0]
        media_path = os.path.join(total_path, filename)
        
        if not os.path.exists(media_path):
            print(f"⚠️ Missing media file: {media_path}, skipping.")
            continue
        
        print(f"\n🎬 [{survey_id} - {base_question}_{submission_type}] Processing: {filename}")
        
        if submission_type in ['AUD', 'VID']:
            # Transcribe audio/video
            transcript = transcribe_media_file(media_path, model)
            if transcript != "[NO AUDIO]":
                # Score with GPT
                question_text = QUESTION_TEXT_MAP[base_question]
                clean_transcript, score, flag = score_transcript_with_gpt(transcript, question_text, client)
            else:
                clean_transcript, score, flag = transcript, 0, "yes"
        else:
            # For TXT files, we'll handle them separately when processing data files
            clean_transcript = "[TEXT_SUBMISSION]"
            score = 100  # Text submissions are considered perfect
            flag = "no"
        
        submissions_by_question[base_question][survey_id][submission_type].append({
            'file': base_name,
            'transcript': clean_transcript,
            'score': score,
            'flag': flag,
            'path': media_path
        })

# 9) process data files to get text submissions
print("\n📄 Processing text submissions from data files...")
text_submissions_found = 0

# Look for Excel files in both root and nested directories
excel_files = []
for root, dirs, files in os.walk(DATA_FILES_ROOT):
    for file in files:
        if file.endswith('.xlsx') and not file.startswith('~'):
            excel_files.append(os.path.join(root, file))

print(f"   - Found {len(excel_files)} Excel files to process")
for excel_file in excel_files:
    print(f"     * {excel_file}")

for excel_path in excel_files:
    # Extract survey_id from the Excel file path
    if excel_path.endswith('_Transcribed.xlsx'):
        survey_id = os.path.basename(excel_path).replace('_Transcribed.xlsx', '')
    else:
        survey_id = os.path.basename(excel_path).replace('.xlsx', '')
    
    print(f"📊 Processing data file: {os.path.basename(excel_path)}")
    
    try:
        df = pd.read_excel(excel_path)
        print(f"   - Data file has {len(df)} rows and {len(df.columns)} columns")
        
        # Look for text submission columns
        txt_columns = [col for col in df.columns if col.endswith('_TXT')]
        print(f"   - Found {len(txt_columns)} TXT columns: {txt_columns}")
        
        for col in txt_columns:
            base_question = col.replace('_TXT', '')
            print(f"   - Processing column: {col} (question: {base_question})")
            
            if base_question in QUESTION_TEXT_MAP:
                # Initialize if needed
                if base_question not in submissions_by_question:
                    submissions_by_question[base_question] = {}
                if survey_id not in submissions_by_question[base_question]:
                    submissions_by_question[base_question][survey_id] = {'TXT': [], 'AUD': [], 'VID': []}
                
                # Process text submissions
                col_submissions = 0
                for idx, text_value in df[col].items():
                    if pd.notna(text_value) and str(text_value).strip():
                        submissions_by_question[base_question][survey_id]['TXT'].append({
                            'file': f"text_{idx}",
                            'transcript': str(text_value).strip(),
                            'score': 100,
                            'flag': "no",
                            'path': f"data_file_{idx}"
                        })
                        col_submissions += 1
                        text_submissions_found += 1
                
                print(f"     * Found {col_submissions} text submissions for {base_question}")
            else:
                print(f"     * Question {base_question} not in QUESTION_TEXT_MAP, skipping")
                
    except Exception as e:
        print(f"❌ Error processing {excel_path}: {e}")

print(f"\n📊 Text processing summary:")
print(f"   - Total text submissions found: {text_submissions_found}")
print(f"   - Questions with submissions: {list(submissions_by_question.keys())}")

# 10) combine submissions and create results
results = []
for base_question, surveys in submissions_by_question.items():
    for survey_id, submission_types in surveys.items():
        # Get all submission types
        txt_submissions = submission_types.get('TXT', [])
        aud_submissions = submission_types.get('AUD', [])
        vid_submissions = submission_types.get('VID', [])
        
        # Process each submission type separately and create individual results
        # This ensures we capture all submissions rather than trying to merge them
        
        # Process text submissions (row-based)
        for txt_sub in txt_submissions:
            results.append({
                "survey_id": survey_id,
                f"{base_question}_File": txt_sub['file'],
                f"{base_question}_Type": "TXT",
                base_question: txt_sub['transcript'],
                f"{base_question}_Score": txt_sub['score'],
                f"{base_question}_Flag": txt_sub['flag']
            })
        
        # Process audio submissions (file-based)
        for aud_sub in aud_submissions:
            results.append({
                "survey_id": survey_id,
                f"{base_question}_File": aud_sub['file'],
                f"{base_question}_Type": "AUD",
                base_question: aud_sub['transcript'],
                f"{base_question}_Score": aud_sub['score'],
                f"{base_question}_Flag": aud_sub['flag']
            })
        
        # Process video submissions (file-based)
        for vid_sub in vid_submissions:
            results.append({
                "survey_id": survey_id,
                f"{base_question}_File": vid_sub['file'],
                f"{base_question}_Type": "VID",
                base_question: vid_sub['transcript'],
                f"{base_question}_Score": vid_sub['score'],
                f"{base_question}_Flag": vid_sub['flag']
            })
        
        # Also create combined results for each respondent (if we can match them)
        # For now, we'll create separate entries for each submission type
        # This gives you maximum flexibility in analysis

# 11) compile into flat results DataFrame
results_df = pd.DataFrame(results)
merged_dfs = []

# Check if we have any results
if results_df.empty:
    print("⚠️ No submissions were processed. This could be because:")
    print("   - Zip files contain only empty media_list.txt files")
    print("   - No text submissions found in data files")
    print("   - No media files found in the extracted folders")
    print("\n📋 Summary of what was found:")
    
    # Check what folders were processed
    processed_folders = []
    for folder in os.listdir(VIDEOS_ROOT):
        folder_path = os.path.join(VIDEOS_ROOT, folder)
        if os.path.isdir(folder_path) and not folder.endswith('.zip'):
            processed_folders.append(folder)
    
    print(f"   - Processed {len(processed_folders)} folders in videos directory")
    for folder in processed_folders:
        print(f"     * {folder}")
    
    # Check data files
    data_files = [f for f in os.listdir(DATA_FILES_ROOT) if f.endswith('.xlsx')]
    print(f"   - Found {len(data_files)} Excel files in data files directory")
    for file in data_files:
        print(f"     * {file}")
    
    print("\n💡 To fix this issue:")
    print("   1. Ensure zip files contain actual media files (not just empty media_list.txt)")
    print("   2. Check that data files contain text submissions in columns ending with _TXT")
    print("   3. Verify that folder names follow the expected naming convention")
    
    # Create empty output files to prevent further errors
    empty_df = pd.DataFrame(columns=['survey_id', 'message'])
    empty_df.to_excel("transcription_results.xlsx", index=False)
    print("\n✅ Created empty transcription_results.xlsx file")
    exit(0)

# 12) for each survey, merge into its data file and collect trimmed DataFrames
for survey_id in results_df["survey_id"].unique():
    # Find data file (may be in subfolder)
    data_file = None
    survey_id_str = str(survey_id)
    
    # Search for data file in DATA_FILES_ROOT and subdirectories
    for root, dirs, files in os.walk(DATA_FILES_ROOT):
        for file in files:
            if file.endswith('.xlsx') and not file.startswith('~') and not file.endswith('_Transcribed.xlsx'):
                if survey_id_str in file or survey_id_str in root:
                    data_file = os.path.join(root, file)
                    break
        if data_file:
            break
    
    if not data_file:
        print(f"⚠️ No data file for survey {survey_id}, skipping merge.")
        continue
    print(f"🔁 Merging into data file: {os.path.basename(data_file)}")

    base_df = pd.read_excel(data_file)
    sub_df = results_df[results_df["survey_id"] == survey_id]
    print(f"   - Base data: {len(base_df)} rows, {len(base_df.columns)} columns")
    print(f"   - Transcriptions: {len(sub_df)} results")

    # Create lookup dictionaries for each question
    # Key: filename (without extension), Value: {transcript, score, flag}
    lookup_dict = {}
    
    for qlabel in QUESTION_TEXT_MAP.keys():
        lookup_dict[qlabel] = {}
        
        # Get all transcriptions for this question
        question_transcriptions = sub_df[sub_df[f"{qlabel}_File"].notna()] if f"{qlabel}_File" in sub_df.columns else pd.DataFrame()
        
        for idx, row in question_transcriptions.iterrows():
            filename = str(row[f"{qlabel}_File"]) if pd.notna(row.get(f"{qlabel}_File")) else None
            if filename:
                lookup_dict[qlabel][filename] = {
                    'transcript': row.get(qlabel, ""),
                    'score': row.get(f"{qlabel}_Score", 0),
                    'flag': row.get(f"{qlabel}_Flag", "no"),
                    'type': row.get(f"{qlabel}_Type", "")
                }
    
    print(f"   - Created lookup dictionary for {len([k for k, v in lookup_dict.items() if v])} questions")
    
    # Now process each question and add merged columns using VLOOKUP approach
    columns_added = 0
    
    for qlabel in QUESTION_TEXT_MAP.keys():
        # Create merged column
        merged_col = f"{qlabel}_Merged"
        score_col = f"{qlabel}_Merged_Score"
        flag_col = f"{qlabel}_Merged_Flag"
        
        # Initialize merged columns
        base_df[merged_col] = ""
        base_df[score_col] = 0
        base_df[flag_col] = "no"
        
        # Process each row
        for idx in range(len(base_df)):
            merged_value = ""
            merged_score = 0
            merged_flag = "no"
            
            # Check TXT column first
            txt_col = f"{qlabel}_TXT"
            if txt_col in base_df.columns:
                txt_value = base_df.loc[idx, txt_col]
                if pd.notna(txt_value) and str(txt_value).strip():
                    # TXT responses go straight to merged
                    merged_value = str(txt_value).strip()
                    merged_score = 100  # TXT responses don't have scores
                    merged_flag = "no"
            
            # Check AUDc1 column if no TXT (use c1 for the code)
            aud_c1_col = f"{qlabel}_AUDc1"
            if not merged_value and aud_c1_col in base_df.columns:
                aud_code = base_df.loc[idx, aud_c1_col]
                if pd.notna(aud_code) and str(aud_code).strip():
                    # Look up the transcription using the code
                    code_str = str(aud_code).strip()
                    if code_str in lookup_dict.get(qlabel, {}):
                        transcription_data = lookup_dict[qlabel][code_str]
                        merged_value = transcription_data['transcript']
                        merged_score = transcription_data['score']
                        merged_flag = transcription_data['flag']
            
            # Check VIDc1 column if no TXT or AUD (use c1 for the code)
            vid_c1_col = f"{qlabel}_VIDc1"
            if not merged_value and vid_c1_col in base_df.columns:
                vid_code = base_df.loc[idx, vid_c1_col]
                if pd.notna(vid_code) and str(vid_code).strip():
                    # Look up the transcription using the code
                    code_str = str(vid_code).strip()
                    if code_str in lookup_dict.get(qlabel, {}):
                        transcription_data = lookup_dict[qlabel][code_str]
                        merged_value = transcription_data['transcript']
                        merged_score = transcription_data['score']
                        merged_flag = transcription_data['flag']
            
            # Set the merged values
            base_df.loc[idx, merged_col] = merged_value
            base_df.loc[idx, score_col] = merged_score
            base_df.loc[idx, flag_col] = merged_flag
        
        # Check if we added any values
        non_empty = base_df[merged_col].str.strip().ne('').sum()
        if non_empty > 0:
            print(f"   - Added {qlabel}_Merged ({non_empty} responses)")
            columns_added += 3
    
    print(f"   - Total merged columns added: {columns_added}")
    
    # Insert merged columns in the right positions (after _VIDc2 or _AUDc2 or _TXT)
    new_column_order = []
    for col in base_df.columns:
        new_column_order.append(col)
        
        # After each question's last column, insert merged columns
        for qlabel in QUESTION_TEXT_MAP.keys():
            vid_c2_col = f"{qlabel}_VIDc2"
            aud_c2_col = f"{qlabel}_AUDc2"
            txt_col = f"{qlabel}_TXT"
            
            # Insert after VIDc2 if it exists, otherwise after AUDc2, otherwise after TXT
            should_insert = False
            if col == vid_c2_col:
                should_insert = True
            elif vid_c2_col not in base_df.columns and col == aud_c2_col:
                should_insert = True
            elif vid_c2_col not in base_df.columns and aud_c2_col not in base_df.columns and col == txt_col:
                should_insert = True
            
            if should_insert:
                # Insert merged columns here
                merged_col = f"{qlabel}_Merged"
                if merged_col in base_df.columns and merged_col not in new_column_order:
                    new_column_order.append(merged_col)
                    new_column_order.append(f"{qlabel}_Merged_Score")
                    new_column_order.append(f"{qlabel}_Merged_Flag")
    
    # Reorder columns
    base_df = base_df[new_column_order]
    print(f"   - Reordered columns with merged columns inserted")

    # add survey_id to base_df
    base_df['survey_id'] = survey_id

    # trim to only response related columns and respondents
    file_cols = [f"{qlabel}_File" for qlabel in QUESTION_TEXT_MAP.keys() if f"{qlabel}_File" in base_df.columns]
    q_cols = []
    for qlabel in QUESTION_TEXT_MAP.keys():
        if qlabel in base_df.columns:
            q_cols.extend([qlabel, f"{qlabel}_Score", f"{qlabel}_Flag"])

    cols_to_keep = ['survey_id'] + file_cols + q_cols
    available_cols = [col for col in cols_to_keep if col in base_df.columns]
    trimmed_df = base_df[available_cols]
    
    # Filter to only rows with responses
    if file_cols:
        mask = trimmed_df[file_cols].apply(lambda row: any(val and str(val).strip() for val in row), axis=1)
        trimmed_df = trimmed_df[mask]

    # save new transcribed data file, do not overwrite original
    transcribed_file = os.path.join(DATA_FILES_ROOT, f"{survey_id}_Transcribed.xlsx")
    try:
        base_df.to_excel(transcribed_file, index=False)
        print(f"✅ Saved Transcribed File: {os.path.basename(transcribed_file)}")
    except PermissionError:
        alt = os.path.join(DATA_FILES_ROOT, f"{survey_id}_Transcribed_copy.xlsx")
        base_df.to_excel(alt, index=False)
        print(f"⚠️ File locked, saved to {os.path.basename(alt)}")

    merged_dfs.append(trimmed_df)

# 13) write global flat results
output_raw = "transcription_results.xlsx"
try:
    results_df.to_excel(output_raw, index=False)
    print(f"\n✅ Global results saved to -> {output_raw}")
except PermissionError:
    alt = "transcription_results_copy.xlsx"
    results_df.to_excel(alt, index=False)
    print(f"⚠️ Global file locked, saved to -> {alt}")

# 14) if any merged DataFrames exist, concat and write respondent-level file
if merged_dfs:
    combined_df = pd.concat(merged_dfs, ignore_index=True)
    output_resp = "transcription_results_respondents.xlsx"
    try:
        combined_df.to_excel(output_resp, index=False)
        print(f"\n✅ Respondent-level results saved to -> {output_resp}")
    except PermissionError:
        alt = "transcription_results_respondents_copy.xlsx"
        combined_df.to_excel(alt, index=False)
        print(f"⚠️ Respondent file locked, saved to -> {alt}")
else:
    print("⚠️ No data files found, skipped respondent-level merge.")
