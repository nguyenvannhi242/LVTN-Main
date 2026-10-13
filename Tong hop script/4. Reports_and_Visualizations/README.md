# Báo cáo Thực nghiệm & Trực quan hóa (Reports & Visualizations)

Thư mục này lưu trữ toàn bộ các tài liệu báo cáo thực nghiệm chi tiết, bảng số liệu và các biểu đồ
trực quan được tạo ra từ các thực nghiệm. Các file trong thư mục này được sử dụng để chèn trực tiếp
vào luận văn tốt nghiệp (Chương 4) và bài báo khoa học (Section IV).

Nội dung chia làm 2 nhóm:
- **Báo cáo văn bản** (3 files .md): Chứa bảng số liệu, phân tích và nhận xét học thuật
- **Biểu đồ trực quan** (7 files .png + 1 file .html): Đồ thị, sơ đồ phục vụ chèn hình vào báo cáo


## Danh sách file (11 files)

```
4. Reports_and_Visualizations/
│
│  ── Báo cáo văn bản ──
├── ensemble_thesis_report.md           # 7 KB  — Báo cáo thực nghiệm mới nhất (bài báo)
├── Feature_Extractor_Report.md         # 6 KB  — Mô tả kỹ thuật 32 đặc trưng
├── optimal_k_report.md                 # 4 KB  — Kết quả lọc đặc trưng tối ưu
│
│  ── Biểu đồ thực nghiệm Hybrid & Alpha ──
├── accuracy_vs_alpha.png               # 180 KB — Grid Search tìm α tối ưu
├── dl_vs_ml_vs_hybrid.png              # 146 KB — So sánh ML vs DL vs Hybrid
│
│  ── Biểu đồ so sánh mô hình ML ──
├── ml_model_comparison.png             # 137 KB — So sánh LightGBM/XGBoost/RF
│
│  ── Biểu đồ SHAP (Giải thích mô hình) ──
├── shap_summary_bar.png                # 246 KB — SHAP global importance
├── shap_summary_beeswarm.png           # 574 KB — SHAP per-sample impact
│
│  ── Biểu đồ OOD ──
├── ood_ai_detection_rates.png          # 511 KB — Detection Rate theo LLM
│
│  ── Biểu đồ lọc đặc trưng ──
├── optimal_k_search.png                # 224 KB — F1-score vs k đặc trưng
└── feature_selection_pipeline.png      # 217 KB — Sơ đồ khối pipeline 3 tầng
```


================================================================================

## Chi tiết từng file


### BÁO CÁO VĂN BẢN


#### ensemble_thesis_report.md
- **Loại**: Báo cáo Markdown (7 KB, 71 dòng)
- **Tạo bởi**: run_ensemble_hybrid_experiment.py
- **Vai trò**: BÁO CÁO THỰC NGHIỆM MỚI NHẤT — chứa số liệu chính thức của bài báo khoa học
- **Mô tả chi tiết**:
  Tổng hợp toàn bộ kết quả thực nghiệm cập nhật từ việc nâng cấp nhánh DL lên 10-Fold Ensemble
  và nhánh ML lên LightGBM mới (huấn luyện trên 32 đặc trưng tối ưu). Nội dung bao gồm:

  **Bảng 4.4 — Hiệu năng Hybrid ID (α*=0.50)**:
  So sánh Accuracy/F1 giữa LightGBM Standalone, GraphCodeBERT Ensemble, và Hybrid Model
  trên 3 tập test nội miền (basic_test, oop_test, unified_test).
  Kết quả đỉnh: Hybrid đạt 95.90% trên unified_test.

  **Bảng 4.5 — Hiệu năng OOD tổng thể (39,999 files)**:
  So sánh 4 phương pháp (LightGBM Cũ, LightGBM Mới, DL Ensemble, Hybrid) trên OOD C và C++.
  Kết quả nổi bật: LightGBM Mới tăng từ 22.53% → 88.01% trên OOD C++ (Domain Adaptation).

  **Bảng 4.5b — TNR/Specificity trên mẫu Human OOD (500 files)**:
  Đánh giá khả năng không báo sai (false positive) trên mã Human ngoại miền.
  Hybrid đạt TNR 90.40% (OOD C) và 86.00% (OOD C++).

  **Bảng 4.6 & 4.7 — Detection Rate theo từng LLM**:
  Tỷ lệ phát hiện AI cho từng mô hình LLM (10 models) trên OOD C và C++.
  Mô hình dễ phát hiện nhất: GPT-4o-mini (99.4% OOD C, 99.5% OOD C++).
  Mô hình khó phát hiện nhất: Gemini-2.0-Pro-Exp (72.2% OOD C).

  **Phân tích học thuật**:
  Nhận xét về hiện tượng định tuyến sai (Misrouting) trong Router Heuristic, phân tích
  nguyên nhân và đề xuất hướng cải tiến (Learned Router).

- **Tương ứng trong luận văn**: Bảng 4.4, 4.5, 4.5b, 4.6, 4.7 (Chương 4.6-4.7)
- **Tương ứng trong bài báo**: Bảng XIV, XV, XVI, XVII, XVIII (Section IV)


#### Feature_Extractor_Report.md
- **Loại**: Báo cáo Markdown (6 KB, 88 dòng)
- **Vai trò**: Tài liệu kỹ thuật mô tả 32 đặc trưng tĩnh (phục vụ Chương 3 — Phương pháp)
- **Mô tả chi tiết**:
  Báo cáo liệt kê và mô tả chi tiết toàn bộ 32 đặc trưng trong bộ CppFeatureExtractorV8,
  được tổ chức thành 5 nhóm chính:

  **Nhóm A — Layout & Formatting** (7 đặc trưng):
  comment_ratio, empty_line_ratio, avg_line_length, max_line_length, tab_vs_space_ratio,
  trailing_space_ratio, brace_style_consistency.
  Ý nghĩa: Phản ánh phong cách trình bày, thói quen thụt lề và sử dụng khoảng trắng.
  AI thường cực kỳ nhất quán, còn Human thường lộn xộn và tùy hứng.

  **Nhóm B — Naming Conventions** (5 đặc trưng):
  avg_identifier_length, identifier_length_variance, single_char_var_ratio,
  unique_identifier_ratio, keyword_to_identifier_ratio.
  Ý nghĩa: Cách đặt tên biến — sinh viên hay dùng biến 1 ký tự (i, j, x), AI đặt tên dài
  và mô tả (total_count, sum_of_array).

  **Nhóm C — Structural Complexity** (5 đặc trưng):
  avg_cyclomatic_complexity, max_nesting_depth, function_count, avg_function_length,
  halstead_difficulty.
  Ý nghĩa: Độ phức tạp cấu trúc — sinh viên nhồi nhét vào main(), AI refactor thành
  nhiều hàm nhỏ.

  **Nhóm D — Habits & Idioms** (5 đặc trưng):
  total_includes, macro_count, has_bits_stdc, has_fast_io, using_namespace_std.
  Ý nghĩa: Thói quen đặc thù — sinh viên dùng bits/stdc++.h, ios::sync_with_stdio,
  macro #define ll long long; AI thường include chính xác và tránh macro.

  **Nhóm E — Information Theory** (5 đặc trưng):
  shannon_entropy, bigram_entropy, whitespace_entropy, token_entropy,
  identifier_entropy.
  Ý nghĩa: Mức độ "hỗn loạn" trong mã nguồn — AI sinh code có entropy khoảng trắng
  thấp (rất đều đặn), entropy tên biến cao (đa dạng từ vựng).

  Ngoài ra báo cáo giải thích:
  - Bộ chỉ số Halstead (Volume, Difficulty, Effort, Estimated Bugs, Time to Program)
  - Chỉ số Maintainability Index (MI) và các đặc trưng OOP
  - Công thức toán học cho từng nhóm

- **Tương ứng trong luận văn**: Bảng 3.1 (danh mục đặc trưng) và Mục 3.3 (Chương 3)


#### optimal_k_report.md
- **Loại**: Báo cáo Markdown (4 KB, 64 dòng)
- **Tạo bởi**: find_optimal_k.py
- **Vai trò**: Kết quả thực nghiệm lọc đặc trưng (phục vụ Chương 3 — Phương pháp)
- **Mô tả chi tiết**:
  Báo cáo trình bày kết quả chạy find_optimal_k.py trên 8,880 records. Nội dung chính:

  **Bảng Grid Search k (Phương pháp 1)**:
  Bảng 20 dòng, mỗi dòng chứa: giá trị k, số đặc trưng sau Lasso L1, F1-score của
  LightGBM/XGBoost/RandomForest/LogisticRegression, và chỉ số Overfit (Train-Val F1 gap).

  Kết quả chính từ bảng:
  - k=5: F1 = 0.8694 (thấp, thiếu thông tin)
  - k=15: F1 = 0.9207 (tăng nhanh)
  - k=33 → 32 đặc trưng sau Lasso: F1 = 0.9458 (đỉnh LightGBM)
  - k=37-40: F1 ≈ 0.944 (bão hòa/giảm nhẹ, overfitting tăng)

  **RFECV (Phương pháp 2)**:
  Recursive Feature Elimination with Cross-Validation (5-Fold) cho kết quả tham chiếu:
  29 đặc trưng tối ưu, F1=0.9440 (Val), F1=0.9548 (Test kín).

  **Kết luận**: k=33 (→ 32 đặc trưng thực tế qua Lasso L1) là cấu hình chuẩn cho hệ thống.
  Tăng k vượt 33 không cải thiện hiệu năng nhưng tăng nguy cơ overfitting (gap Train-Val
  F1 tăng từ 0.0516 lên 0.0552).

- **Tương ứng trong luận văn**: Bảng 3.2 và Hình 10 (Chương 3)


================================================================================

### BIỂU ĐỒ THỰC NGHIỆM


#### accuracy_vs_alpha.png (180 KB)
- **Tạo bởi**: run_ensemble_hybrid_experiment.py (Phần G)
- **Tương ứng**: Hình 12 trong luận văn, Figure 5 trong bài báo
- **Mô tả**:
  Đồ thị đường (line plot) thể hiện kết quả Grid Search tìm hệ số lai α tối ưu.
  - Trục x: Giá trị α (từ 0.00 đến 1.00, bước 0.02)
  - Trục y: Accuracy (%) trên tập unified_test (1,000 mẫu)
  - Điểm đỉnh: α* = 0.50, Accuracy = 95.90%
  - Hai đầu mút: α=0 (ML-only, ~87.9%), α=1 (DL-only, ~92.9%)
  Đồ thị hình parabol lõm cho thấy kết hợp lai luôn tốt hơn từng nhánh riêng lẻ.


#### dl_vs_ml_vs_hybrid.png (146 KB)
- **Tạo bởi**: run_ensemble_hybrid_experiment.py (Phần G)
- **Tương ứng**: Hình 11 trong luận văn
- **Mô tả**:
  Đồ thị cột nhóm (grouped bar chart) so sánh trực quan Accuracy giữa 3 phương pháp:
  - Cột xanh: LightGBM (ML only)
  - Cột cam: GraphCodeBERT Ensemble (DL only)
  - Cột đỏ/tím: Hybrid Model (đề xuất)
  Mỗi nhóm 3 cột tương ứng với 1 tập test (basic_test, oop_test, unified_test).
  Cho thấy Hybrid cân bằng ưu điểm: ML mạnh trên Basic, DL mạnh trên OOP,
  Hybrid mạnh trên cả hai.


#### ml_model_comparison.png (137 KB)
- **Tạo bởi**: retrain_improved.py hoặc run_chapter4_experiments.py
- **Tương ứng**: Hình 13 trong luận văn (Thực nghiệm 1 — So sánh mô hình ML)
- **Mô tả**:
  Đồ thị cột nhóm so sánh Accuracy giữa các mô hình học máy cổ điển:
  - LightGBM Cũ (44 đặc trưng)
  - LightGBM Mới (32 đặc trưng, retrained)
  - XGBoost
  - Random Forest
  Mỗi nhóm tương ứng với 1 tập test (basic_test, oop_test, unified_test).
  Cho thấy LightGBM Mới (32 đặc trưng) đạt hiệu năng tương đương hoặc tốt hơn
  LightGBM Cũ (44 đặc trưng), chứng minh pipeline lọc 3 tầng không mất thông tin
  hữu ích mà còn giảm nhiễu.


#### shap_summary_bar.png (246 KB)
- **Tạo bởi**: retrain_improved.py (phần SHAP)
- **Tương ứng**: Hình 14 trong luận văn (Thực nghiệm 2 — XAI bằng SHAP)
- **Mô tả**:
  Biểu đồ cột ngang SHAP Summary (Bar plot). Thể hiện mean |SHAP value| (trung bình
  trị tuyệt đối giá trị SHAP) của top 20 đặc trưng ảnh hưởng lớn nhất đến mô hình
  LightGBM.

  Cách đọc: Đặc trưng ở trên cùng = quan trọng nhất. Giá trị SHAP càng lớn = đặc
  trưng đó có ảnh hưởng càng mạnh đến quyết định phân loại AI/Human.

  Top 5 đặc trưng quan trọng nhất (ước tính):
  1. whitespace_entropy — Entropy khoảng trắng
  2. avg_identifier_length — Độ dài trung bình tên biến
  3. brace_style_consistency — Nhất quán kiểu ngoặc nhọn
  4. comment_ratio — Tỷ lệ chú thích
  5. tab_vs_space_ratio — Tỷ lệ Tab/Space


#### shap_summary_beeswarm.png (574 KB)
- **Tạo bởi**: retrain_improved.py (phần SHAP)
- **Tương ứng**: Hình 15 trong luận văn (Thực nghiệm 2 — XAI bằng SHAP)
- **Mô tả**:
  Biểu đồ Beeswarm SHAP — biểu đồ phân tích sâu nhất về mô hình.

  Cách đọc:
  - Mỗi chấm = 1 mẫu dữ liệu (1 file mã nguồn)
  - Vị trí ngang: SHAP value (dương = đẩy về phía dự đoán AI, âm = đẩy về phía Human)
  - Màu sắc: Giá trị thực của đặc trưng (đỏ = giá trị cao, xanh = giá trị thấp)

  Ví dụ cách diễn giải:
  - whitespace_entropy: Chấm xanh (giá trị thấp) nằm bên phải (SHAP dương)
    → Entropy khoảng trắng THẤP → Dự đoán AI (AI viết code đều đặn, ít ngẫu nhiên)
  - avg_identifier_length: Chấm đỏ (giá trị cao) nằm bên phải (SHAP dương)
    → Tên biến DÀI → Dự đoán AI (AI đặt tên mô tả đầy đủ kiểu clean code)
  - single_char_var_ratio: Chấm đỏ (giá trị cao) nằm bên trái (SHAP âm)
    → Nhiều biến 1 ký tự → Dự đoán Human (sinh viên hay dùng i, j, n, x)


#### ood_ai_detection_rates.png (511 KB)
- **Tạo bởi**: run_ensemble_hybrid_experiment.py (Phần G)
- **Tương ứng**: Hình 16 trong luận văn, Figure 6 trong bài báo (Thực nghiệm 5)
- **Mô tả**:
  Biểu đồ cột/nhóm trực quan hóa AI Detection Rate (tỷ lệ phát hiện mã AI) trên tập
  OOD cho từng mô hình LLM sinh mã. Chia riêng OOD C và OOD C++.

  Cho thấy:
  - GPT-4o-mini luôn dễ phát hiện nhất (Detection Rate >99%)
  - Gemini-2.0-Pro-Exp khó phát hiện nhất (Detection Rate ~70%)
  - Hybrid Model (cột đỏ/tím) luôn bằng hoặc cao hơn từng nhánh riêng lẻ


#### optimal_k_search.png (224 KB)
- **Tạo bởi**: find_optimal_k.py
- **Tương ứng**: Hình 10 trong luận văn (Chương 3 — Phương pháp)
- **Mô tả**:
  Đồ thị đường thể hiện F1-score (trục y) của 4 mô hình ML theo số lượng đặc trưng
  đầu vào k (trục x, từ 5 đến 40).

  Các đường:
  - LightGBM (xanh dương): F1 tăng dần, đỉnh tại k=33
  - XGBoost (cam): Xu hướng tương tự LightGBM
  - Random Forest (xanh lá): F1 thấp hơn, tăng chậm hơn
  - Logistic Regression (đỏ): F1 thấp nhất, bão hòa sớm

  Đồ thị cho thấy hiệu năng tăng nhanh từ k=5-15, tăng chậm từ k=15-33, rồi bão hòa
  sau k=33. Điểm k=33 (→ 32 đặc trưng sau Lasso) là cân bằng tối ưu.


#### feature_selection_pipeline.png (217 KB)
- **Tạo bởi**: visualize_pipeline.py
- **Tương ứng**: Hình 9 trong luận văn (Chương 3 — Phương pháp)
- **Mô tả**:
  Sơ đồ khối (block diagram) mô tả trực quan kiến trúc pipeline lọc đặc trưng 3 tầng:

  ```
  [44 đặc trưng gốc]
       │
       ▼
  ┌─────────────────────────┐
  │ Tầng 1: Correlation     │
  │ Filter (|r| > 0.85)     │ → Loại đặc trưng tương quan cao
  └─────────────────────────┘
       │
       ▼
  ┌─────────────────────────┐
  │ Tầng 2: SelectKBest     │
  │ (Mutual Information)    │ → Chọn top k=33 đặc trưng
  └─────────────────────────┘
       │
       ▼
  ┌─────────────────────────┐
  │ Tầng 3: Lasso L1        │
  │ (Logistic Regression)   │ → Ép trọng số dư thừa về 0
  └─────────────────────────┘
       │
       ▼
  [32 đặc trưng tối ưu]
  ```

  Dùng chèn vào Chương 3 (Phương pháp) để minh họa quy trình.


================================================================================

## Mapping biểu đồ ↔ Hình trong luận văn

| File | Hình trong luận văn | Thực nghiệm |
|:---|:---|:---|
| feature_selection_pipeline.png | Hình 9 | Chương 3 (Phương pháp) |
| optimal_k_search.png | Hình 10 | Chương 3 (Phương pháp) |
| dl_vs_ml_vs_hybrid.png | Hình 11 | TN4 — So sánh ML/DL/Hybrid |
| accuracy_vs_alpha.png | Hình 12 | TN4 — Grid Search α |
| ml_model_comparison.png | Hình 13 | TN1 — So sánh mô hình ML |
| shap_summary_bar.png | Hình 14 | TN2 — SHAP global importance |
| shap_summary_beeswarm.png | Hình 15 | TN2 — SHAP per-sample |
| ood_ai_detection_rates.png | Hình 16 | TN5 — OOD Detection Rate |
