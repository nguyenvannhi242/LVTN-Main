import os
import json
import hashlib
import random
import re
import shutil
from collections import Counter
# Paths
WORKSPACE = r"C:\Users\2\Desktop\WORKSPACE\LVTN-Main"
TRAIN_SET_DIR = os.path.join(WORKSPACE, "data_splits", "train_set")
TEST_SET_ID_DIR = os.path.join(WORKSPACE, "data_splits", "test_set_ID")
TEST_SET_OOD_DIR = os.path.join(WORKSPACE, "data_splits", "test_set_OOD")
TEST_SET_UNIFIED_DIR = os.path.join(WORKSPACE, "data_splits", "test_set_unified")

def get_hash(text):
    normalized = "".join(text.split())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def strip_metadata_headers(code_raw):
    lines = code_raw.split('\n')
    cleaned_lines = []
    header_passed = False
    for line in lines:
        if not header_passed:
            if line.strip().startswith('//') and any(kw in line.upper() for kw in ['DATASET', 'MODEL', 'GEMINI', 'GPT', 'CATEGORY']):
                continue
            if line.strip() != '':
                header_passed = True
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def make_record(code, label, file_name, relative_path, project_name, source=None):
    num_lines = len(code.split('\n'))
    num_chars = len(code)
    ext = os.path.splitext(file_name)[1]
    
    rec = {
        "code": code,
        "language": "C++",
        "label": label,
        "file_name": file_name,
        "file_extension": ext,
        "relative_path": relative_path.replace("\\", "/"),
        "project_name": project_name,
        "num_lines": num_lines,
        "num_chars": num_chars
    }
    if source:
        rec['source'] = source
    return rec

def scan_dir_files(path):
    files_found = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(('.cpp', '.c', '.h', '.hpp', '.CPP', '.C', '.H', '.HPP')):
                files_found.append(os.path.join(root, file))
    return files_found

def main():
    print("=== STARTING IMPROVED DATASET INTEGRATION AND SPLIT ===")
    random.seed(42)
    
    # -------------------------------------------------------------------------
    # STEP 1: LOAD ORIGINAL TRAIN SET & GET HASHES
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading original train set for strict deduplication...")
    orig_train_file = os.path.join(TRAIN_SET_DIR, "train_data_with_oop.jsonl")
    backup_train_file = os.path.join(TRAIN_SET_DIR, "train_data_with_oop_backup.jsonl")
    
    # Backup original train file if not already backed up
    if not os.path.exists(backup_train_file):
        print(f"  Creating backup of train set: {backup_train_file}")
        shutil.copy2(orig_train_file, backup_train_file)
        
    orig_train_records = []
    train_hashes = set()
    with open(backup_train_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            orig_train_records.append(data)
            train_hashes.add(get_hash(data.get("code", "")))
            
    print(f"  Loaded original train set size: {len(orig_train_records)}")
    print(f"  Loaded original train hashes  : {len(train_hashes)}")

    # -------------------------------------------------------------------------
    # STEP 2: PROCESS OOD AND SELECT SAMPLES FOR TRAINING
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Processing OOD C and CPP datasets for training split...")
    
    ood_train_additions = []
    
    for lang in ["C", "CPP"]:
        lang_dir = os.path.join(TEST_SET_OOD_DIR, lang)
        subdirs = sorted([d for d in os.listdir(lang_dir) if os.path.isdir(os.path.join(lang_dir, d))])
        
        lang_train_added = 0
        lang_test_count = 0
        
        ood_test_records = []
        
        for sd in subdirs:
            sd_path = os.path.join(lang_dir, sd)
            label = "HUMAN" if sd.lower() == 'human' else "AI"
            
            # Find subfolders (Normal/Paraphrased for AI, nan for Human)
            subfolders = sorted([sf for sf in os.listdir(sd_path) if os.path.isdir(os.path.join(sd_path, sf))])
            
            for sf in subfolders:
                sf_path = os.path.join(sd_path, sf)
                files_found = scan_dir_files(sf_path)
                
                # Load all files
                records = []
                for file_path in files_found:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            code_raw = f.read()
                        cleaned_code = strip_metadata_headers(code_raw)
                        file_name = os.path.basename(file_path)
                        rel_path = os.path.relpath(file_path, TEST_SET_OOD_DIR)
                        h = get_hash(cleaned_code)
                        records.append({
                            'record': make_record(cleaned_code, label, file_name, rel_path, file_name, source=sd),
                            'hash': h,
                            'path': file_path
                        })
                    except Exception:
                        continue
                
                # Shuffle records
                random.shuffle(records)
                
                # Determine how many to take for training
                if label == "HUMAN":
                    # We have 1 folder 'nan' with 1,000 files. We take 500 for training.
                    take_n = 500
                else:
                    # AI has 2 folders ('Normal', 'Paraphrased'). Each has 1,000 files.
                    # We take 25 from each folder, total 250 AI files per language (10 models * 2 styles * 25 = 500).
                    take_n = 25
                    
                train_part = records[:take_n]
                test_part = records[take_n:]
                
                # Add to train
                for r in train_part:
                    ood_train_additions.append(r['record'])
                    train_hashes.add(r['hash'])
                    lang_train_added += 1
                    
                # Add to test
                for r in test_part:
                    ood_test_records.append(r['record'])
                    lang_test_count += 1
                    
            print(f"    Processed {lang} - {sd:<30} | Train: {lang_train_added} | Test: {lang_test_count}")
            lang_train_added = 0
            lang_test_count = 0
            
        # Write OOD test jsonl file
        ood_jsonl_path = os.path.join(TEST_SET_OOD_DIR, f"ood_{lang.lower()}_test.jsonl")
        random.shuffle(ood_test_records)
        with open(ood_jsonl_path, 'w', encoding='utf-8') as f:
            for r in ood_test_records:
                f.write(json.dumps(r) + '\n')
        print(f"  Saved OOD {lang} test set to: {ood_jsonl_path} with {len(ood_test_records)} records.")

    # -------------------------------------------------------------------------
    # STEP 3: CREATE NEW MERGED TRAIN SET
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Creating new merged training set...")
    new_train_records = orig_train_records + ood_train_additions
    random.shuffle(new_train_records)
    
    new_train_file = os.path.join(TRAIN_SET_DIR, "train_data_with_oop.jsonl")
    with open(new_train_file, 'w', encoding='utf-8') as f:
        for r in new_train_records:
            f.write(json.dumps(r) + '\n')
            
    print(f"  Saved new train set to: {new_train_file} | Total records: {len(new_train_records)}")
    train_counts = Counter(r['label'] for r in new_train_records)
    print(f"  New train set balance: {dict(train_counts)}")

    # -------------------------------------------------------------------------
    # STEP 4: REGENERATE IN-DISTRIBUTION (ID) TEST SETS WITH ZERO LEAKAGE
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Regenerating ID test sets (basic_test.jsonl and oop_test.jsonl) with zero leakage...")
    
    # 1. basic_test.jsonl
    basic_ai_dir = os.path.join(TEST_SET_ID_DIR, "Basic_C++", "AI-Generated")
    basic_human_dir = os.path.join(TEST_SET_ID_DIR, "Basic_C++", "Human-written")
    
    # AI files: Balance across the 10 models (25 files each = 250 AI files)
    basic_ai_models = [d for d in os.listdir(basic_ai_dir) if os.path.isdir(os.path.join(basic_ai_dir, d))]
    ai_subfolders = []
    for m in basic_ai_models:
        m_path = os.path.join(basic_ai_dir, m)
        if m == "FromAI-v1":
            ai_subfolders.append(m_path)
        elif m == "FromAI-v2":
            for subm in os.listdir(m_path):
                subm_path = os.path.join(m_path, subm)
                if os.path.isdir(subm_path):
                    ai_subfolders.append(subm_path)
                    
    basic_ai_selected = []
    print("  Sampling clean AI files for Basic ID:")
    for path in sorted(ai_subfolders):
        subfolder_name = os.path.basename(path)
        parent_folder = os.path.basename(os.path.dirname(path))
        display_name = f"{parent_folder}/{subfolder_name}" if parent_folder == "FromAI-v2" else subfolder_name
        
        clean_in_subfolder = []
        for root, _, files in os.walk(path):
            for file in files:
                if not file.endswith(('.cpp', '.c', '.h', '.hpp')):
                    continue
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code_raw = f.read()
                    cleaned_code = strip_metadata_headers(code_raw)
                    h = get_hash(cleaned_code)
                    if h not in train_hashes:
                        clean_in_subfolder.append((file_path, file, cleaned_code))
                except Exception:
                    continue
                    
        print(f"    - {display_name:<30}: {len(clean_in_subfolder)} clean files.")
        selected = random.sample(clean_in_subfolder, min(25, len(clean_in_subfolder)))
        basic_ai_selected.extend(selected)
        
    # Human files: Sample 250 files
    basic_human_clean = []
    for root, _, files in os.walk(basic_human_dir):
        for file in files:
            if not file.endswith(('.cpp', '.c', '.h', '.hpp')):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code_raw = f.read()
                cleaned_code = strip_metadata_headers(code_raw)
                h = get_hash(cleaned_code)
                if h not in train_hashes:
                    basic_human_clean.append((file_path, file, cleaned_code))
            except Exception:
                continue
                
    print(f"  Total Basic Human clean files: {len(basic_human_clean)}")
    basic_human_selected = random.sample(basic_human_clean, min(250, len(basic_human_clean)))
    
    # Save basic_test.jsonl
    basic_test_file = os.path.join(TEST_SET_ID_DIR, "basic_test.jsonl")
    basic_records = []
    for file_path, file, code in basic_ai_selected:
        rel_path = os.path.relpath(file_path, TEST_SET_ID_DIR)
        basic_records.append(make_record(code, "AI", file, rel_path, file))
    for file_path, file, code in basic_human_selected:
        rel_path = os.path.relpath(file_path, TEST_SET_ID_DIR)
        basic_records.append(make_record(code, "HUMAN", file, rel_path, file))
        
    random.shuffle(basic_records)
    with open(basic_test_file, 'w', encoding='utf-8') as f:
        for r in basic_records:
            f.write(json.dumps(r) + '\n')
    print(f"  Saved basic_test.jsonl with {len(basic_records)} records.")

    # 2. oop_test.jsonl
    oop_ai_dir = os.path.join(TEST_SET_ID_DIR, "OOP_C++", "AI-Generated")
    oop_human_dir = os.path.join(TEST_SET_ID_DIR, "OOP_C++", "Human-written")
    
    oop_ai_clean = []
    for root, _, files in os.walk(oop_ai_dir):
        for file in files:
            if not file.endswith(('.cpp', '.c', '.h', '.hpp')):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code_raw = f.read()
                cleaned_code = strip_metadata_headers(code_raw)
                h = get_hash(cleaned_code)
                if h not in train_hashes:
                    oop_ai_clean.append((file_path, file, cleaned_code))
            except Exception:
                continue
                
    print(f"  Total OOP AI clean files: {len(oop_ai_clean)}")
    oop_ai_selected = random.sample(oop_ai_clean, min(250, len(oop_ai_clean)))
    
    oop_human_clean = []
    for root, _, files in os.walk(oop_human_dir):
        for file in files:
            if not file.endswith(('.cpp', '.c', '.h', '.hpp')):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code_raw = f.read()
                cleaned_code = strip_metadata_headers(code_raw)
                h = get_hash(cleaned_code)
                if h not in train_hashes:
                    oop_human_clean.append((file_path, file, cleaned_code))
            except Exception:
                continue
                
    print(f"  Total OOP Human clean files: {len(oop_human_clean)}")
    oop_human_selected = random.sample(oop_human_clean, min(250, len(oop_human_clean)))
    
    oop_test_file = os.path.join(TEST_SET_ID_DIR, "oop_test.jsonl")
    oop_records = []
    for file_path, file, code in oop_ai_selected:
        rel_path = os.path.relpath(file_path, TEST_SET_ID_DIR)
        oop_records.append(make_record(code, "AI", file, rel_path, file))
    for file_path, file, code in oop_human_selected:
        rel_path = os.path.relpath(file_path, TEST_SET_ID_DIR)
        oop_records.append(make_record(code, "HUMAN", file, rel_path, file))
        
    random.shuffle(oop_records)
    with open(oop_test_file, 'w', encoding='utf-8') as f:
        for r in oop_records:
            f.write(json.dumps(r) + '\n')
    print(f"  Saved oop_test.jsonl with {len(oop_records)} records.")

    # 3. unified_test.jsonl
    unified_records = basic_records + oop_records
    random.shuffle(unified_records)
    unified_test_file = os.path.join(TEST_SET_UNIFIED_DIR, "unified_test.jsonl")
    with open(unified_test_file, 'w', encoding='utf-8') as f:
        for r in unified_records:
            f.write(json.dumps(r) + '\n')
    print(f"  Saved unified_test.jsonl with {len(unified_records)} records.")

    # -------------------------------------------------------------------------
    # STEP 5: VERIFICATION PIPELINE
    # -------------------------------------------------------------------------
    print("\n=== RUNNING DATASET VERIFICATION PIPELINE (ZERO LEAKAGE) ===")
    
    def verify(filepath, expected_count, name, check_balance=True):
        if not os.path.exists(filepath):
            print(f"  [FAIL] {name} does not exist!")
            return False
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        print(f"  {name} ({os.path.basename(filepath)}):")
        print(f"    - Total records: {len(records)} (Expected: {expected_count})")
        if len(records) != expected_count and expected_count is not None:
            print(f"    - [FAIL] Count mismatch!")
            return False
            
        ai_cnt = sum(1 for r in records if r.get('label') == 'AI')
        human_cnt = sum(1 for r in records if r.get('label') == 'HUMAN')
        print(f"    - Label distribution: AI={ai_cnt}, HUMAN={human_cnt}")
        if check_balance and ai_cnt != human_cnt:
            print(f"    - [FAIL] Imbalance found!")
            return False
            
        leakage_cnt = 0
        for r in records:
            h = get_hash(r.get('code', ''))
            if h in train_hashes:
                leakage_cnt += 1
        print(f"    - Leakage check: {leakage_cnt} training file hashes matched.")
        if leakage_cnt > 0:
            print(f"    - [FAIL] Leakage detected! Test set contains training data.")
            return False
        print("    - [OK] Verified.")
        return True

    success = True
    success &= verify(basic_test_file, 500, "Basic Test (ID)")
    success &= verify(oop_test_file, 500, "OOP Test (ID)")
    success &= verify(unified_test_file, 1000, "Unified Test")
    
    # OOD expected counts:
    # C OOD has: AI = 2000 * 10 - 25 * 20 = 19500; HUMAN = 1000 - 500 = 500. Total = 20,000.
    # CPP OOD has: AI = 2000 * 10 - 2 (deepseek-r1 Paraphrased) - 25 * 20 = 19498; HUMAN = 1000 - 500 = 500. Total = 19,998.
    success &= verify(os.path.join(TEST_SET_OOD_DIR, "ood_c_test.jsonl"), 20000, "OOD C Test", check_balance=False)
    success &= verify(os.path.join(TEST_SET_OOD_DIR, "ood_cpp_test.jsonl"), 19999, "OOD CPP Test", check_balance=False)
    
    if success:
        print("\n[SUCCESS] ALL DATASETS REGENERATED AND VERIFIED SUCCESSFULLY WITH ZERO LEAKAGE!")
    else:
        print("\n[FAIL] SOME DATASETS FAILED VERIFICATION. Check errors above.")

if __name__ == "__main__":
    main()
