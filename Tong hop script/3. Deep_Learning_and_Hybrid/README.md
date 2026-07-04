# Mô hình Học sâu & Mô hình Lai (Deep Learning & Hybrid)

Thư mục này chứa toàn bộ mã nguồn thực nghiệm liên quan đến **nhánh Deep Learning** (mô hình ngôn ngữ
mã nguồn GraphCodeBERT) và **thuật toán kết hợp lai** (Hybrid Routing Model) — phương pháp đề xuất chính
của luận văn.

Mô hình Hybrid kết hợp ưu điểm của cả 2 nhánh:
- Nhánh ML (LightGBM): Mạnh về phát hiện mã AI dạng Basic C++ nhờ 32 đặc trưng tĩnh
- Nhánh DL (GraphCodeBERT): Mạnh về phát hiện mã AI dạng OOP C++ nhờ hiểu ngữ nghĩa code

Công thức tổ hợp: P_hybrid = α × P_BERT + (1 − α) × P_LGBM


## Danh sách file (3 files)

```
3. Deep_Learning_and_Hybrid/
├── Basic_of_LVTN_Graphcodebert_tính_micro_&_macro.ipynb   # 2.2 MB — Huấn luyện 10-fold DL
├── Hybrid_model.ipynb                                      # 511 KB — Thực nghiệm Hybrid v1
└── run_ensemble_hybrid_experiment.py                       # 54 KB  — Thực nghiệm toàn diện v2
```


================================================================================

## Chi tiết từng file


### Basic_of_LVTN_Graphcodebert_tính_micro_&_macro.ipynb
- **Loại**: Jupyter Notebook (2.2 MB — chứa cả output cells)
- **Nguồn gốc**: Google Colab (chạy trên GPU T4/A100)
- **Vai trò**: Huấn luyện mô hình học sâu GraphCodeBERT (Thực nghiệm 3, Chương 4)
- **Mô tả chi tiết**:
  Notebook CHÍNH huấn luyện mô hình học sâu GraphCodeBERT (microsoft/graphcodebert-base) trên
  bộ dữ liệu phát hiện mã AI. Đây là nhánh Deep Learning của hệ thống Hybrid.

  **Kiến trúc mô hình**:
  - Base model: microsoft/graphcodebert-base (125M parameters)
  - Tokenizer: RobertaTokenizer, max_length=512 tokens
  - Head: Linear layer (768 → 2) cho phân loại nhị phân AI/Human
  - Kỹ thuật xử lý mã dài: Sliding window chunking (stride=256, max_len=512)
    Với mỗi file mã nguồn dài hơn 512 tokens, chia thành nhiều chunk chồng lấn,
    chạy inference trên từng chunk rồi lấy trung bình xác suất.

  **Phương pháp kiểm thử chéo 10-fold Leave-One-Group-Out (LOGO)**:
  Dữ liệu được chia thành 10 nhóm (groups) theo mô hình AI sinh mã:
  - Fold 1: Test trên GPT-4o, train trên 9 LLMs còn lại
  - Fold 2: Test trên Claude-3.5, train trên 9 LLMs còn lại
  - ... (tương tự cho DeepSeek, Gemini, Llama, Qwen, O3-mini, v.v.)

  Phương pháp LOGO đảm bảo đánh giá cross-model generalization — mô hình phải phát hiện
  được mã từ một LLM mà nó CHƯA TỪNG thấy trong quá trình huấn luyện. Đây là tiêu chuẩn
  đánh giá khắt khe hơn nhiều so với random split thông thường.

  **Cấu hình huấn luyện**:
  - Optimizer: AdamW, learning rate = 2e-5, weight decay = 0.01
  - Epochs: 3 epochs/fold (tổng 30 epochs cho 10 fold)
  - Batch size: 16 (train), 32 (eval)
  - Mixed precision: FP16 (trên GPU hỗ trợ)

  **Đầu ra của mỗi fold**:
  - Model weights (.pt hoặc .bin) lưu tại DL models/
  - File predictions JSON chứa xác suất dự đoán cho từng mẫu test
  - Chỉ số: Accuracy, Macro F1, Micro F1, Precision, Recall, Confusion Matrix

  **Kết quả tổng hợp 10-fold**:
  - Macro F1 trung bình = 0.9290 (chỉ tính riêng nhánh DL)
  - Mô hình khó phát hiện nhất: DeepSeek-R1 (F1 thấp nhất)
  - Mô hình dễ phát hiện nhất: GPT-4o-mini (F1 cao nhất)

  **Tính toán Macro F1 và Micro F1**:
  Notebook chứa phần code riêng giải thích và tính thủ công (step-by-step) sự khác biệt
  giữa Macro F1 (trung bình F1 của 2 class AI/Human — công bằng khi class imbalance) và
  Micro F1 (tổng hợp TP/FP/FN toàn cục — phản ánh hiệu năng tổng thể).

- **Yêu cầu**: torch, transformers, GPU (bắt buộc cho huấn luyện, khuyến nghị A100/T4)
- **Dữ liệu**: train_data_with_oop.jsonl (~8,880 records)
- **Đầu ra**: 10 model GraphCodeBERT + 10 file predictions JSON


### Hybrid_model.ipynb
- **Loại**: Jupyter Notebook (511 KB)
- **Nguồn gốc**: Google Colab
- **Vai trò**: Thực nghiệm lai ghép BẢN ĐẦU (Thực nghiệm 4, Chương 4 — phiên bản cũ)
- **Mô tả chi tiết**:
  Notebook thực nghiệm kết hợp lai (Hybrid Routing Model) phiên bản đầu tiên. Notebook thực hiện:

  1. **Load predictions**: Đọc file xác suất dự đoán từ GraphCodeBERT 1-Fold (chỉ 1 fold,
     không phải Ensemble) và LightGBM Standalone (model cũ với 44 đặc trưng).
  2. **Công thức kết hợp lai**:
     P_hybrid(α) = α × P_BERT + (1 − α) × P_LGBM
     Nếu P_hybrid ≥ 0.5 → dự đoán AI, ngược lại → dự đoán Human.
  3. **Grid Search tìm alpha tối ưu**: Quét α từ 0.0 đến 1.0, bước 0.01, đánh giá
     Accuracy trên unified_test. Kết quả: α = 0.48.
  4. **So sánh hiệu năng**: Vẽ bảng và biểu đồ so sánh ML-only vs DL-only vs Hybrid
     trên 3 tập test ID.

  **LƯU Ý QUAN TRỌNG — ĐÂY LÀ BẢN CŨ**:
  - Sử dụng GraphCodeBERT 1-Fold (không phải 10-fold Ensemble)
  - Hệ số lai: α = 0.48
  - Đỉnh hiệu năng unified_test: 93.80%
  - Tập OOD: 12,599 files

  Bản MỚI (bài báo khoa học) sử dụng run_ensemble_hybrid_experiment.py với:
  - GraphCodeBERT 10-fold Ensemble → xác suất mượt mà hơn
  - Hệ số lai: α* = 0.50
  - Đỉnh hiệu năng unified_test: 95.90%
  - Tập OOD: 39,999 files

- **Dữ liệu**: Predictions JSON từ GraphCodeBERT + LightGBM
- **Đầu ra**: Bảng so sánh hiệu năng + biểu đồ alpha


### run_ensemble_hybrid_experiment.py
- **Loại**: Python Script (54 KB, ~600 dòng)
- **Nguồn gốc**: Thư mục `tools/`
- **Vai trò**: MÃ NGUỒN CHẠY THỰC NGHIỆM CHÍNH CỦA BÀI BÁO KHOA HỌC (BẢN MỚI NHẤT)
- **Mô tả chi tiết**:
  Script toàn diện nhất trong toàn bộ dự án, tích hợp tất cả các thí nghiệm cập nhật cho
  bài báo khoa học. Script được chia thành 7 phần chính:

  **PHẦN A — Tổ hợp Ensemble DL (10-Fold Averaging)**:
  Đọc 10 file predictions JSON (mỗi file chứa xác suất dự đoán từ 1 fold GraphCodeBERT).
  Tính xác suất trung bình Ensemble:
    S_DL = (1/10) × Σ_{i=1}^{10} P_fold_i
  Phương pháp Ensemble giúp giảm phương sai dự đoán (variance reduction) và tăng tính
  ổn định so với dùng 1 fold đơn lẻ. Giải thích trực giác: nếu 1 fold bị "nhầm" trên
  1 mẫu, 9 fold còn lại có thể "bù lại" → xác suất trung bình đáng tin cậy hơn.

  **PHẦN B — Grid Search tìm Alpha tối ưu**:
  Quét α từ 0.00 đến 1.00 (bước 0.02) trên tập unified_test (1,000 mẫu):
    P_hybrid(α) = α × S_DL + (1 − α) × S_ML
  Tại mỗi giá trị α, tính Accuracy và F1-score. Vẽ đồ thị Accuracy vs Alpha.
  Kết quả: α* = 0.50 cho hiệu năng cao nhất:
    - Accuracy: 95.90%
    - F1-score: 0.9590
  Ý nghĩa α=0.50: Nhánh DL và nhánh ML đóng góp NGANG BẰNG nhau — cho thấy 2 nhánh
  bổ trợ (complementary) chứ không dư thừa (redundant).

  **PHẦN C — Đánh giá In-Distribution (ID)**:
  Chạy Hybrid (α=0.50) trên 3 tập test ID:
  | Tập test     | LightGBM | GraphCodeBERT Ensemble | Hybrid |
  |-------------|----------|----------------------|--------|
  | basic_test   | 93.20%   | 86.00%               | 93.80% |
  | oop_test     | 82.60%   | 99.80%               | 98.00% |
  | unified_test | 87.90%   | 92.90%               | 95.90% |

  **PHẦN D — Đánh giá Out-of-Distribution (OOD) trên 39,999 files**:
  Chạy trên 2 tập OOD hoàn toàn mới, không trùng phân phối với dữ liệu huấn luyện:
  - OOD C: 20,000 files (10,000 AI từ 10 LLMs + 10,000 Human)
  - OOD C++: 19,999 files (~10,000 AI từ 10 LLMs + ~10,000 Human)
  Dữ liệu lấy từ CodeMirage, bao gồm mã từ: GPT-4o-mini, DeepSeek-R1, DeepSeek-V3,
  Claude-3.5-Haiku, Gemini-2.0-Flash, Gemini-2.0-Flash-Thinking, Gemini-2.0-Pro,
  Llama3.3-70B, O3-mini, Qwen2.5-Coder.

  **PHẦN E — Đánh giá Thích nghi miền (Domain Adaptation)**:
  So sánh trực tiếp:
  - LightGBM Cũ (chưa thích nghi): OOD C 50.09%, OOD C++ **22.53%**
  - LightGBM Mới (retrain 2,000 mẫu đại diện): OOD C 87.11%, OOD C++ **88.01%**
  Kết quả cho thấy chiến lược thích nghi miền (retrain trên một tập mẫu OOD nhỏ đại diện)
  giúp LightGBM tăng vọt hiệu năng trên dữ liệu ngoại miền, đặc biệt trên OOD C++
  (tăng gần 66 điểm phần trăm).

  **PHẦN F — Phân tích Detection Rate theo từng LLM**:
  Tính tỷ lệ phát hiện mã AI (AI Detection Rate) cho từng mô hình LLM trên cả tập
  OOD C và C++. Xuất bảng chi tiết tương ứng với Bảng XVI, XVII, XVIII trong bài báo.

  **PHẦN G — Trực quan hóa**:
  Tự động vẽ 3 biểu đồ:
  - accuracy_vs_alpha.png: Đồ thị Grid Search
  - dl_vs_ml_vs_hybrid.png: So sánh cột 3 phương pháp
  - ood_ai_detection_rates.png: Detection Rate theo LLM
  Các biểu đồ được lưu tại 4. Reports_and_Visualizations/.

- **Yêu cầu**: torch, transformers, lightgbm, shap, scikit-learn, matplotlib
- **Dữ liệu**: 5 tập JSONL + 10 file predictions từ GraphCodeBERT
- **Đầu ra**: Console report + 3 biểu đồ PNG + ensemble_thesis_report.md


================================================================================

## So sánh 2 phiên bản thực nghiệm

| Thuộc tính | Hybrid_model.ipynb (CŨ) | run_ensemble_hybrid_experiment.py (MỚI) |
|:---|:---|:---|
| Nhánh DL | GraphCodeBERT 1-Fold | GraphCodeBERT 10-Fold Ensemble |
| Nhánh ML | LightGBM Cũ (44 đặc trưng) | LightGBM Mới (32 đặc trưng, retrained) |
| Hệ số α tối ưu | 0.48 | 0.50 |
| Đỉnh Accuracy (unified) | 93.80% | 95.90% |
| Tập OOD | 12,599 files | 39,999 files |
| Domain Adaptation | Không | Có (LightGBM retrained 2,000 mẫu) |
| Dùng cho | Luận văn (bản nộp ban đầu) | Bài báo khoa học (bản cập nhật) |
