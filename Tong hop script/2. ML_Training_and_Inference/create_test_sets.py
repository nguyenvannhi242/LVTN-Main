import os
import json
import hashlib
import random
import re

def get_hash(text):
    # Normalize whitespace to make comparison robust to minor formatting changes
    normalized = "".join(text.split())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def strip_metadata_headers(code_raw):
    """ Remove potential metadata leakage headers from AI code """
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

def make_record(code, label, file_name, relative_path, project_name):
    # Calculate stats
    num_lines = len(code.split('\n'))
    num_chars = len(code)
    ext = os.path.splitext(file_name)[1]
    
    return {
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

def main():
    print("=== STARTING DATASET SPLIT GENERATION ===")
    random.seed(42)
    
    workspace = r"C:\Users\2\Desktop\WORKSPACE\LVTN-Main"
    train_set_dir = os.path.join(workspace, "data_splits", "train_set")
    test_set_id_dir = os.path.join(workspace, "data_splits", "test_set_ID")
    test_set_ood_dir = os.path.join(workspace, "data_splits", "test_set_OOD")
    test_set_unified_dir = os.path.join(workspace, "data_splits", "test_set_unified")
    
    # -------------------------------------------------------------------------
    # STEP 1: LOAD DEDUPLICATION DATABASE
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading training sets for strict deduplication...")
    
    # Load basic train hashes
    train_basic_file = os.path.join(train_set_dir, "train_data_10_model_AI.jsonl")
    train_basic_hashes = set()
    with open(train_basic_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            train_basic_hashes.add(get_hash(data.get("code", "")))
            
    # Load combined train hashes to avoid ML training leakage in OOP test
    train_with_oop_file = os.path.join(train_set_dir, "train_data_with_oop.jsonl")
    train_combined_hashes = set()
    with open(train_with_oop_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            train_combined_hashes.add(get_hash(data.get("code", "")))
            
    print(f"  Loaded basic training hashes: {len(train_basic_hashes)}")
    print(f"  Loaded combined ML training hashes (with OOP): {len(train_combined_hashes)}")
    
    # -------------------------------------------------------------------------
    # STEP 2: CREATE BASIC C++ TEST SET (500 SAMPLES)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Creating Basic C++ test set (basic_test.jsonl)...")
    
    basic_ai_dir = os.path.join(test_set_id_dir, "Basic_C++", "AI-Generated")
    basic_human_dir = os.path.join(test_set_id_dir, "Basic_C++", "Human-written")
    
    # AI files: Balance across the 10 models (25 files each = 250 AI files)
    basic_ai_models = [d for d in os.listdir(basic_ai_dir) if os.path.isdir(os.path.join(basic_ai_dir, d))]
    # Wait, let's handle nested FromAI-v2 folders
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
                    
    print(f"  Found AI model subfolders: {len(ai_subfolders)}")
    
    basic_ai_selected = []
    # We want 25 files per subfolder (since there are 10 subfolders: FromAI-v1, and 9 in FromAI-v2)
    # Let's count them
    print("  Sampling from AI subfolders:")
    for path in sorted(ai_subfolders):
        subfolder_name = os.path.basename(path)
        parent_folder = os.path.basename(os.path.dirname(path))
        display_name = f"{parent_folder}/{subfolder_name}" if parent_folder == "FromAI-v2" else subfolder_name
        
        # Scan clean files in this subfolder
        clean_in_subfolder = []
        for root, _, files in os.walk(path):
            for file in files:
                if not file.endswith(('.cpp', '.c', '.h', '.hpp')):
                    continue
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code_raw = f.read()
                except Exception:
                    continue
                
                h = get_hash(code_raw)
                if h not in train_basic_hashes and h not in train_combined_hashes:
                    clean_in_subfolder.append((file_path, file, code_raw))
                    
        print(f"    - {display_name:<30}: {len(clean_in_subfolder)} clean files found.")
        # Sample exactly 25 files
        selected = random.sample(clean_in_subfolder, min(25, len(clean_in_subfolder)))
        basic_ai_selected.extend(selected)
        
    print(f"  Total Basic AI selected: {len(basic_ai_selected)} (Expected: 250)")
    
    # Human files: Sample 250 files from basic_human_dir
    basic_human_clean = []
    for root, _, files in os.walk(basic_human_dir):
        for file in files:
            if not file.endswith(('.cpp', '.c', '.h', '.hpp')):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code_raw = f.read()
            except Exception:
                continue
                
            h = get_hash(code_raw)
            if h not in train_basic_hashes and h not in train_combined_hashes:
                basic_human_clean.append((file_path, file, code_raw))
                
    print(f"  Total Basic Human clean files found: {len(basic_human_clean)}")
    basic_human_selected = random.sample(basic_human_clean, min(250, len(basic_human_clean)))
    print(f"  Total Basic Human selected: {len(basic_human_selected)} (Expected: 250)")
    
    # Save basic_test.jsonl
    basic_test_file = os.path.join(test_set_id_dir, "basic_test.jsonl")
    basic_records = []
    
    for file_path, file, code in basic_ai_selected:
        cleaned_code = strip_metadata_headers(code)
        rel_path = os.path.relpath(file_path, test_set_id_dir)
        basic_records.append(make_record(cleaned_code, "AI", file, rel_path, file))
        
    for file_path, file, code in basic_human_selected:
        cleaned_code = strip_metadata_headers(code)
        rel_path = os.path.relpath(file_path, test_set_id_dir)
        basic_records.append(make_record(cleaned_code, "HUMAN", file, rel_path, file))
        
    # Shuffle for fairness
    random.shuffle(basic_records)
    
    with open(basic_test_file, 'w', encoding='utf-8') as f:
        for r in basic_records:
            f.write(json.dumps(r) + '\n')
            
    print(f"  Saved Basic test set to: {basic_test_file}")
    
    # -------------------------------------------------------------------------
    # STEP 3: CREATE OOP C++ TEST SET (500 SAMPLES)
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Creating OOP C++ test set (oop_test.jsonl)...")
    
    oop_ai_dir = os.path.join(test_set_id_dir, "OOP_C++", "AI-Generated")
    oop_human_dir = os.path.join(test_set_id_dir, "OOP_C++", "Human-written")
    
    # Scan all OOP AI files that are NOT in combined ML train hashes
    oop_ai_clean = []
    for root, _, files in os.walk(oop_ai_dir):
        for file in files:
            if not file.endswith(('.cpp', '.c', '.h', '.hpp')):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code_raw = f.read()
            except Exception:
                continue
                
            h = get_hash(code_raw)
            if h not in train_combined_hashes:
                oop_ai_clean.append((file_path, file, code_raw))
                
    print(f"  Total Unused OOP AI clean files found: {len(oop_ai_clean)}")
    oop_ai_selected = random.sample(oop_ai_clean, min(250, len(oop_ai_clean)))
    print(f"  Total OOP AI selected: {len(oop_ai_selected)} (Expected: 250)")
    
    # Scan all OOP Human files that are NOT in combined ML train hashes
    oop_human_clean = []
    for root, _, files in os.walk(oop_human_dir):
        for file in files:
            if not file.endswith(('.cpp', '.c', '.h', '.hpp')):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code_raw = f.read()
            except Exception:
                continue
                
            h = get_hash(code_raw)
            if h not in train_combined_hashes:
                oop_human_clean.append((file_path, file, code_raw))
                
    print(f"  Total Unused OOP Human clean files found: {len(oop_human_clean)}")
    oop_human_selected = random.sample(oop_human_clean, min(250, len(oop_human_clean)))
    print(f"  Total OOP Human selected: {len(oop_human_selected)} (Expected: 250)")
    
    # Save oop_test.jsonl
    oop_test_file = os.path.join(test_set_id_dir, "oop_test.jsonl")
    oop_records = []
    
    for file_path, file, code in oop_ai_selected:
        cleaned_code = strip_metadata_headers(code)
        rel_path = os.path.relpath(file_path, test_set_id_dir)
        oop_records.append(make_record(cleaned_code, "AI", file, rel_path, file))
        
    for file_path, file, code in oop_human_selected:
        cleaned_code = strip_metadata_headers(code)
        rel_path = os.path.relpath(file_path, test_set_id_dir)
        oop_records.append(make_record(cleaned_code, "HUMAN", file, rel_path, file))
        
    random.shuffle(oop_records)
    
    with open(oop_test_file, 'w', encoding='utf-8') as f:
        for r in oop_records:
            f.write(json.dumps(r) + '\n')
            
    print(f"  Saved OOP test set to: {oop_test_file}")
    
    # -------------------------------------------------------------------------
    # STEP 4: CREATE UNIFIED TEST SET (1,000 SAMPLES)
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Merging Basic and OOP test sets to create unified_test.jsonl...")
    
    unified_records = basic_records + oop_records
    random.shuffle(unified_records)
    
    unified_test_file = os.path.join(test_set_unified_dir, "unified_test.jsonl")
    with open(unified_test_file, 'w', encoding='utf-8') as f:
        for r in unified_records:
            f.write(json.dumps(r) + '\n')
            
    print(f"  Saved Unified test set to: {unified_test_file}")
    print(f"  Total unified records: {len(unified_records)}")
    
    # -------------------------------------------------------------------------
    # STEP 5: CREATE OOD TEST SETS (C & CPP)
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Compiling Out-of-Distribution test sets...")
    
    for lang in ["C", "CPP"]:
        lang_dir = os.path.join(test_set_ood_dir, lang)
        print(f"  Processing language: {lang} in {lang_dir}...")
        
        ood_records = []
        subdirs = sorted([d for d in os.listdir(lang_dir) if os.path.isdir(os.path.join(lang_dir, d))])
        
        for sd in subdirs:
            sd_path = os.path.join(lang_dir, sd)
            label = "HUMAN" if sd.lower() == 'human' else "AI"
            
            # Recurse and find C/C++ files
            files_found = []
            for root, _, files in os.walk(sd_path):
                for file in files:
                    if file.endswith(('.cpp', '.c', '.h', '.hpp', '.CPP', '.C', '.H', '.HPP')):
                        files_found.append(os.path.join(root, file))
                        
            print(f"    - Source: {sd:<30} | Files: {len(files_found)}")
            
            for file_path in files_found:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code_raw = f.read()
                except Exception:
                    continue
                
                cleaned_code = strip_metadata_headers(code_raw)
                file_name = os.path.basename(file_path)
                rel_path = os.path.relpath(file_path, test_set_ood_dir)
                
                # We can store extra metadata like 'source' in the OOD records if needed, but keeping standard schema
                record = make_record(cleaned_code, label, file_name, rel_path, file_name)
                # Attach extra tags for OOD analysis
                record['source'] = sd
                ood_records.append(record)
                
        # Write to JSONL
        random.shuffle(ood_records)
        ood_jsonl = os.path.join(test_set_ood_dir, f"ood_{lang.lower()}_test.jsonl")
        
        with open(ood_jsonl, 'w', encoding='utf-8') as f:
            for r in ood_records:
                f.write(json.dumps(r) + '\n')
                
        print(f"  Saved OOD {lang} test set to: {ood_jsonl} | Count: {len(ood_records)}")
        
    # -------------------------------------------------------------------------
    # STEP 6: SELF-VERIFICATION PIPELINE
    # -------------------------------------------------------------------------
    print("\n=== RUNNING DATASET SELF-VERIFICATION PIPELINE ===")
    
    def verify_file(filepath, expected_count, name, allowed_hashes_blacklist=None):
        print(f"\nVerifying {name} ({filepath}):")
        if not os.path.exists(filepath):
            print(f"  [FAIL] File does not exist!")
            return False
            
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
                
        print(f"  Total records: {len(records)} (Expected: {expected_count})")
        if len(records) != expected_count:
            print(f"  [FAIL] Record count mismatch!")
            return False
            
        # Check label balance
        ai_count = sum(1 for r in records if r.get('label') == 'AI')
        human_count = sum(1 for r in records if r.get('label') == 'HUMAN')
        print(f"  Labels: AI = {ai_count}, HUMAN = {human_count}")
        if ai_count != human_count and expected_count in [500, 1000]:
            print(f"  [FAIL] Label imbalance found!")
            return False
            
        # Check leakage against blacklist
        if allowed_hashes_blacklist:
            leakage_count = 0
            for r in records:
                h = get_hash(r.get('code', ''))
                if h in allowed_hashes_blacklist:
                    leakage_count += 1
            print(f"  Leakage check: {leakage_count} leaked training files found.")
            if leakage_count > 0:
                print(f"  [FAIL] Leakage detected! Clean check failed.")
                return False
        
        print("  [OK] Verification PASSED!")
        return True

    success = True
    success &= verify_file(basic_test_file, 500, "Basic test set", train_basic_hashes)
    success &= verify_file(oop_test_file, 500, "OOP test set", train_combined_hashes)
    success &= verify_file(unified_test_file, 1000, "Unified test set", train_combined_hashes)
    success &= verify_file(os.path.join(test_set_ood_dir, "ood_c_test.jsonl"), 6300, "OOD C test set")
    success &= verify_file(os.path.join(test_set_ood_dir, "ood_cpp_test.jsonl"), 6299, "OOD CPP test set")
    
    if success:
        print("\n[SUCCESS] ALL DATASET GENERATIONS AND VERIFICATIONS COMPLETED SUCCESSFULLY WITH ZERO LEAKAGE!")
    else:
        print("\n[FAIL] SOME VERIFICATIONS FAILED! Please check the errors above.")

if __name__ == "__main__":
    main()
