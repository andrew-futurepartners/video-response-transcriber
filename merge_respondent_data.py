"""
Script to merge transcription_results.xlsx with the original data file
Creates merged columns using VLOOKUP approach:
- TXT: Copy from existing _TXT column
- AUD/VID: Lookup code from existing column, find transcription, put in merged
"""
import os
import pandas as pd
import numpy as np

# Configuration
TRANSCRIPTION_FILE = "transcription_results.xlsx"
DATA_FILES_ROOT = "data files"

# Question text map
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

print("🔄 Starting respondent-level merge (v2 - VLOOKUP approach)...")

# Load transcription results
if not os.path.exists(TRANSCRIPTION_FILE):
    print(f"❌ {TRANSCRIPTION_FILE} not found!")
    exit(1)

print(f"📊 Loading {TRANSCRIPTION_FILE}...")
results_df = pd.read_excel(TRANSCRIPTION_FILE)
print(f"   - Found {len(results_df)} transcription results")

# Find all data files
excel_files = []
for root, dirs, files in os.walk(DATA_FILES_ROOT):
    for file in files:
        if file.endswith('.xlsx') and not file.startswith('~') and not file.endswith('_Transcribed.xlsx'):
            excel_files.append(os.path.join(root, file))

print(f"\n📁 Found {len(excel_files)} data files:")
for excel_file in excel_files:
    print(f"   - {excel_file}")

# Process each survey
for survey_id in results_df["survey_id"].unique():
    # Find matching data file
    data_file = None
    survey_id_str = str(survey_id)
    for excel_path in excel_files:
        if survey_id_str in excel_path:
            data_file = excel_path
            break
    
    if not data_file:
        print(f"⚠️ No data file found for survey {survey_id}, skipping merge.")
        continue
    
    print(f"\n🔁 Merging survey {survey_id_str} with {os.path.basename(data_file)}...")
    
    # Load base data
    base_df = pd.read_excel(data_file)
    print(f"   - Base data has {len(base_df)} rows, {len(base_df.columns)} columns")
    
    # Get transcriptions for this survey
    sub_df = results_df[results_df["survey_id"] == survey_id]
    print(f"   - Found {len(sub_df)} transcription results")
    
    # Create lookup dictionaries for each question and type
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
    
    print(f"   - Created lookup dictionary for {len(lookup_dict)} questions")
    
    # Now process each question and add merged columns
    columns_added = 0
    
    for qlabel in QUESTION_TEXT_MAP.keys():
        # Check which columns exist in the base dataset
        txt_col = f"{qlabel}_TXT"
        aud_col = f"{qlabel}_AUD"
        vid_col = f"{qlabel}_VID"
        
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
    
    # Insert merged columns in the right positions
    # Build new column order - insert after _VIDc2 or _AUDc2 or _TXT
    new_column_order = []
    for col in base_df.columns:
        new_column_order.append(col)
        
        # After each question's last column (VIDc2, or AUDc2 if no VID, or TXT if only TXT), insert merged
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
    
    # Save transcribed data file
    transcribed_file = os.path.join(os.path.dirname(data_file), f"{survey_id_str}_Transcribed.xlsx")
    try:
        base_df.to_excel(transcribed_file, index=False)
        print(f"✅ Saved transcribed file: {os.path.basename(transcribed_file)}")
        print(f"   - Total columns: {len(base_df.columns)}")
    except PermissionError:
        alt = os.path.join(os.path.dirname(data_file), f"{survey_id_str}_Transcribed_copy.xlsx")
        base_df.to_excel(alt, index=False)
        print(f"⚠️ File locked, saved to {os.path.basename(alt)}")

print("\n🎉 Merge complete!")
print("📁 Check the _Transcribed.xlsx file(s) in the data files folder")
