import os
import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import numpy as np

# Configuration
WORKSPACE = r"C:\Users\2\Desktop\WORKSPACE\LVTN-Main"
DL_BASIC_DIR = os.path.join(WORKSPACE, "DL models", "C++_Basic_detection_10_fold_LeaveOneGroupOut_new")
DL_OOP_DIR = os.path.join(WORKSPACE, "DL models", "C++_OOP_detection_10_fold_LeaveOneGroupOut")

class FastInferenceDataset(Dataset):
    def __init__(self, codes, tokenizer, max_len=512, stride=256):
        self.chunks = []
        for idx, code in enumerate(codes):
            # Tokenize without special tokens
            enc = tokenizer(code, add_special_tokens=False)
            tokens = enc['input_ids']
            if not tokens:
                tokens = [tokenizer.unk_token_id]

            # Slide window chunking
            for start in range(0, max(1, len(tokens) - max_len + 3), stride):
                chunk = tokens[start : start + max_len - 2]
                chunk_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]

                pad_len = max_len - len(chunk_ids)
                input_ids = chunk_ids + [tokenizer.pad_token_id] * pad_len
                mask = [1] * len(chunk_ids) + [0] * pad_len

                self.chunks.append((idx,
                                    torch.tensor(input_ids, dtype=torch.long),
                                    torch.tensor(mask, dtype=torch.long)))

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, i):
        return self.chunks[i]

def main():
    print("=== STARTING SANITY CHECK FOR 10-FOLD ENSEMBLE ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load sample codes
    sample_codes = [
        # Basic sample
        "int main() { int a = 5; return 0; }",
        # OOP sample
        "class Test { public: void hello() {} }; int main() { Test t; t.hello(); return 0; }"
    ]
    print(f"Loaded {len(sample_codes)} sample codes for test.")

    # 2. Check DL model directories
    print(f"Basic DL Dir: {DL_BASIC_DIR} (exists: {os.path.exists(DL_BASIC_DIR)})")
    print(f"OOP DL Dir: {DL_OOP_DIR} (exists: {os.path.exists(DL_OOP_DIR)})")

    # Get fold paths
    basic_folds = sorted([os.path.join(DL_BASIC_DIR, d) for d in os.listdir(DL_BASIC_DIR) if os.path.isdir(os.path.join(DL_BASIC_DIR, d)) and d.startswith('fold_')])
    oop_folds = sorted([os.path.join(DL_OOP_DIR, d) for d in os.listdir(DL_OOP_DIR) if os.path.isdir(os.path.join(DL_OOP_DIR, d)) and d.startswith('fold_')])
    print(f"Found {len(basic_folds)} Basic folds and {len(oop_folds)} OOP folds.")

    # 3. Test Tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained(basic_folds[0])
    print("Tokenizer loaded successfully.")

    # 4. Create Dataset and Loader
    dataset = FastInferenceDataset(sample_codes, tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    print(f"Dataset created with {len(dataset)} chunks.")

    # 5. Test sequential model loading and inference
    print("\nTesting Basic model fold 1 and fold 2...")
    for fold_path in basic_folds[:2]:
        print(f"Loading {os.path.basename(fold_path)}...")
        model = AutoModelForSequenceClassification.from_pretrained(fold_path)
        model.to(device).eval()

        chunk_probs = defaultdict(list)
        with torch.no_grad():
            for idxs, input_ids, mask in loader:
                input_ids = input_ids.to(device)
                mask = mask.to(device)
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        logits = model(input_ids, attention_mask=mask).logits
                else:
                    logits = model(input_ids, attention_mask=mask).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                for idx, p in zip(idxs.numpy(), probs):
                    chunk_probs[idx].append(float(p))

        fold_probs = np.array([np.mean(chunk_probs[i]) for i in range(len(sample_codes))])
        print(f"  Fold probabilities: {fold_probs}")

        # Clear VRAM
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    print("\nTesting OOP model fold 1 and fold 2...")
    for fold_path in oop_folds[:2]:
        print(f"Loading {os.path.basename(fold_path)}...")
        model = AutoModelForSequenceClassification.from_pretrained(fold_path)
        model.to(device).eval()

        chunk_probs = defaultdict(list)
        with torch.no_grad():
            for idxs, input_ids, mask in loader:
                input_ids = input_ids.to(device)
                mask = mask.to(device)
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        logits = model(input_ids, attention_mask=mask).logits
                else:
                    logits = model(input_ids, attention_mask=mask).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                for idx, p in zip(idxs.numpy(), probs):
                    chunk_probs[idx].append(float(p))

        fold_probs = np.array([np.mean(chunk_probs[i]) for i in range(len(sample_codes))])
        print(f"  Fold probabilities: {fold_probs}")

        # Clear VRAM
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    print("\n=== SANITY CHECK PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    main()
