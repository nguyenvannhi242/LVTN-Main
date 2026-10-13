# Huấn luyện & Suy luận Mô hình Học máy (ML Training & Inference)

Thư mục này chứa toàn bộ mã nguồn liên quan đến việc **huấn luyện** (training), **đánh giá** (evaluation)
và **suy luận** (inference) của nhánh Machine Learning trong hệ thống phát hiện mã AI. Nhánh ML sử dụng
các mô hình học máy cổ điển (LightGBM, XGBoost, Random Forest, Logistic Regression) hoạt động trên
32 đặc trưng tĩnh được trích xuất bởi bộ CppFeatureExtractorV8.

Thư mục cũng chứa các script tự động hóa thực nghiệm toàn diện (automation scripts) phục vụ việc
tái tạo kết quả trong Chương 4 của luận văn.


## Danh sách file (10 files)

```
2. ML_Training_and_Inference/
├── retrain.py                          # 29 KB — Huấn luyện v1 (44 đặc trưng, sơ khai)
├── retrain_improved.py                 # 25 KB — Huấn luyện v2 (32 đặc trưng, chính thức)
├── create_test_sets.py                 # 17 KB — Tạo tập test v1
├── create_test_sets_improved.py        # 17 KB — Tạo tập test v2 (có OOD)
├── generate_baselines_improved.py      # 13 KB — Tạo baselines cho web UI
├── inference_improved.py               # 17 KB — Suy luận trên 5 tập test (ID + OOD)
├── inference_ood.py                    # 22 KB — Suy luận chuyên sâu OOD (theo từng LLM)
├── run_chapter4_experiments.py         # 68 KB — Tự động hóa thực nghiệm Chương 4 (bản cũ)
├── run_sanity_check_ensemble.py        # 6 KB  — Kiểm tra tính tương thích Ensemble DL
└── visualize_pipeline.py               # 5 KB  — Vẽ biểu đồ pipeline & Feature Importance
```


================================================================================

## Chi tiết từng file


### retrain.py
- **Loại**: Python Script (29 KB, ~600 dòng)
- **Nguồn gốc**: Thư mục `retrain ML models/`
- **Vai trò**: Phiên bản huấn luyện đầu tiên (v1) — đã được thay thế bởi retrain_improved.py
- **Mô tả chi tiết**:
  Script huấn luyện ban đầu, thực hiện quy trình:

  1. **Đọc dữ liệu**: Load file train_data_10_model_AI.jsonl (dữ liệu Basic C++) và
     OOP_cpp_ai_detection_dataset.jsonl (dữ liệu OOP C++).
  2. **Lấy mẫu OOP**: Chọn ngẫu nhiên (random sampling, seed=42) 300 mẫu AI + 300 mẫu Human
     từ dataset OOP để gộp vào tập huấn luyện. Chiến lược này giúp mô hình tiếp xúc với cả
     2 phong cách lập trình (Basic + OOP) mà không bị thiên lệch sang OOP.
  3. **Trích xuất đặc trưng**: Sử dụng CppFeatureExtractorV8 để trích xuất **44 đặc trưng gốc**
     (phiên bản chưa qua pipeline lọc 3 tầng).
  4. **Huấn luyện**: Train 3 mô hình (LightGBM, XGBoost, Random Forest) với hyperparameter mặc
     định, không sử dụng cross-validation.
  5. **Lưu model**: Xuất file .pkl cho từng mô hình.

  **Hạn chế so với retrain_improved.py**: Dùng 44 đặc trưng (dư thừa), không có Stratified K-Fold,
  không tính SHAP, không tích hợp Logistic Regression, không có pipeline lọc đặc trưng tự động.

- **Dữ liệu**: train_data_10_model_AI.jsonl + OOP_cpp_ai_detection_dataset.jsonl
- **Đầu ra**: Các file model .pkl lưu trong saved_models_44_features/


### retrain_improved.py
- **Loại**: Python Script (25 KB, 532 dòng)
- **Nguồn gốc**: Thư mục `retrain ML models/`
- **Vai trò**: Script huấn luyện CHÍNH THỨC (production-ready) — sử dụng trong kết quả luận văn
- **Mô tả chi tiết**:
  Script huấn luyện cải tiến toàn diện, tạo ra các model được dùng trong hệ thống triển khai.
  Các điểm nâng cấp quan trọng so với retrain.py:

  1. **Bộ trích xuất tích hợp**: Định nghĩa trực tiếp class CppFeatureExtractorV8 trong script
     (32 đặc trưng đã qua pipeline lọc 3 tầng). Các module khác (find_optimal_k.py, inference)
     có thể import class này từ đây.
  2. **Huấn luyện 4 mô hình song song**:
     - LightGBM (mô hình chính, được dùng trong Hybrid Model)
     - XGBoost (đối chứng)
     - Random Forest (đối chứng)
     - Logistic Regression (đối chứng đơn giản nhất)
  3. **Stratified K-Fold Cross-Validation** (k=5): Chia dữ liệu thành 5 fold phân tầng
     (đảm bảo tỷ lệ AI:Human đồng đều trong mỗi fold), huấn luyện trên 4 fold và đánh giá
     trên 1 fold, lặp lại 5 lần.
  4. **Tính toán chỉ số chi tiết**: Accuracy, F1-score (Macro & Micro), Precision, Recall,
     Confusion Matrix cho từng fold và trung bình.
  5. **Tích hợp SHAP**: Sau khi huấn luyện xong, script tự động tính giá trị SHAP (SHapley
     Additive exPlanations) cho từng đặc trưng trên tập validation, phục vụ Thực nghiệm 2
     (XAI — Giải thích mô hình) trong Chương 4. Xuất biểu đồ shap_summary_bar.png và
     shap_summary_beeswarm.png.
  6. **Lưu trữ artifacts**: Xuất 3 file quan trọng:
     - Model đã train (.joblib) — hiệu quả hơn .pkl cho scikit-learn
     - scaler.pkl — StandardScaler đã fit trên tập train
     - final_features.pkl — Danh sách 32 tên đặc trưng cuối cùng
  7. **Hàm tiện ích strip_metadata_headers()**: Xóa các header metadata AI (do GPT, Claude, v.v.
     thường chèn vào đầu file) trước khi trích xuất đặc trưng, giúp mô hình không bị lệ thuộc
     vào metadata mà tập trung vào phong cách viết code.

- **Dữ liệu**: data_splits/train_set/train_data_with_oop.jsonl (~8,880 records)
- **Đầu ra**: Thư mục saved_models_44_features_improved/ chứa model, scaler, feature list


### create_test_sets.py
- **Loại**: Python Script (17 KB)
- **Nguồn gốc**: Thư mục `retrain ML models/`
- **Vai trò**: Phiên bản tạo tập dữ liệu test ban đầu (v1)
- **Mô tả chi tiết**:
  Script phân tách dữ liệu thành tập huấn luyện (train) và tập kiểm thử (test) ban đầu. Chỉ
  xử lý dữ liệu Basic C++ từ file train_data_10_model_AI.jsonl. Sử dụng train_test_split của
  scikit-learn với stratified sampling (phân tầng theo nhãn AI/Human) để đảm bảo cân bằng.

  **Hạn chế**: Chưa tạo tập OOP test, unified test, và OOD test. Được thay thế bởi
  create_test_sets_improved.py.

- **Đầu ra**: basic_test.jsonl (phiên bản sơ khai)


### create_test_sets_improved.py
- **Loại**: Python Script (17 KB)
- **Nguồn gốc**: Thư mục `retrain ML models/`
- **Vai trò**: Tạo toàn bộ cấu trúc data_splits/ dùng trong thực nghiệm chính thức
- **Mô tả chi tiết**:
  Script cải tiến, tạo ra đầy đủ các tập dữ liệu kiểm thử theo cấu trúc data_splits/:

  1. **Tập train**: Gộp Basic C++ và OOP C++ thành train_data_with_oop.jsonl
  2. **Tập test ID**:
     - basic_test.jsonl: 500 mẫu Basic C++ (250 AI + 250 Human), holdout split
     - oop_test.jsonl: 500 mẫu OOP C++ (250 AI + 250 Human), holdout split
  3. **Tập unified**: unified_test.jsonl: Gộp basic_test + oop_test (1,000 mẫu)
  4. **Tập OOD**: Tạo ood_c_test.jsonl và ood_cpp_test.jsonl từ dữ liệu CodeMirage
     (đọc các file .c/.cpp thô từ thư mục test_set_OOD/C/ và test_set_OOD/CPP/,
     gộp thành JSONL với metadata: label, model name, language)

  Script đảm bảo KHÔNG có dữ liệu trùng lặp giữa train và test (data leakage prevention).

- **Đầu ra**: Toàn bộ thư mục data_splits/ (xem 5. Datasets/README.md)


### generate_baselines_improved.py
- **Loại**: Python Script (13 KB)
- **Nguồn gốc**: Thư mục `retrain ML models/`
- **Vai trò**: Tạo baselines phục vụ giao diện web ứng dụng
- **Mô tả chi tiết**:
  Script tính giá trị baseline (giá trị trung bình — mean) của từng đặc trưng cho 2 nhóm:
  mã AI và mã Human. Kết quả được lưu thành file baselines.json với cấu trúc:

  ```json
  {
    "comment_ratio": {"ai": 0.042, "human": 0.018},
    "avg_identifier_length": {"ai": 8.73, "human": 5.21},
    ...
  }
  ```

  File baselines.json được sử dụng trong giao diện web Streamlit (app.py) để hiển thị:
  - Radar chart so sánh đặc trưng mã đang phân tích với giá trị trung bình AI và Human
  - Thanh ngang (bar) cho từng đặc trưng với 2 mốc tham chiếu (AI baseline, Human baseline)

  Script xử lý 500 mẫu AI + 500 mẫu Human (random sample, seed=42) từ tập train để tính mean.

- **Dữ liệu**: train_data_10_model_AI.jsonl (lấy mẫu 1000)
- **Đầu ra**: models/zoo/baselines.json


### inference_improved.py
- **Loại**: Python Script (17 KB, ~350 dòng)
- **Nguồn gốc**: Thư mục `retrain ML models/`
- **Vai trò**: Đánh giá mô hình ML trên toàn bộ 5 tập kiểm thử (Thực nghiệm 1, Chương 4)
- **Mô tả chi tiết**:
  Script suy luận (inference) chính thức, chạy mô hình đã huấn luyện trên 5 tập kiểm thử:

  1. **basic_test** (500 mẫu Basic C++)
  2. **oop_test** (500 mẫu OOP C++)
  3. **unified_test** (1,000 mẫu hợp nhất)
  4. **ood_c_test** (20,000 files OOD C)
  5. **ood_cpp_test** (19,999 files OOD C++)

  Với mỗi tập, quy trình xử lý:
  - Đọc file JSONL, trích xuất 32 đặc trưng bằng CppFeatureExtractorV8
  - Chuẩn hóa đặc trưng bằng StandardScaler đã lưu (scaler.pkl)
  - Chạy prediction bằng LightGBM (và tùy chọn XGBoost, Random Forest)
  - Tính và in: Accuracy, F1-score, Precision, Recall, Confusion Matrix, Classification Report

  Script cũng hỗ trợ chạy so sánh đồng thời nhiều mô hình ML, xuất bảng kết quả dạng
  DataFrame phục vụ chèn vào Bảng 4.1 (So sánh các mô hình ML) trong luận văn.

- **Dữ liệu**: 5 file JSONL từ data_splits/
- **Đầu ra**: In kết quả ra console + xuất báo cáo JSON


### inference_ood.py
- **Loại**: Python Script (22 KB, 507 dòng)
- **Nguồn gốc**: Thư mục `retrain ML models/`
- **Vai trò**: Phân tích chi tiết hiệu năng OOD theo từng mô hình LLM (Thực nghiệm 5, Chương 4)
- **Mô tả chi tiết**:
  Script suy luận chuyên biệt cho tập dữ liệu ngoại miền (Out-of-Distribution), cung cấp
  phân tích sâu hơn inference_improved.py:

  1. **Phân tích theo từng mô hình LLM**: Tách kết quả prediction theo từng mô hình AI sinh
     mã (GPT-4o-mini, DeepSeek-R1, DeepSeek-V3, Claude-3.5-Haiku, Gemini-2.0-Flash,
     Gemini-2.0-Flash-Thinking, Gemini-2.0-Pro, Llama3.3-70B, O3-mini, Qwen2.5-Coder) để
     xác định mô hình nào dễ phát hiện nhất và mô hình nào khó nhất.
  2. **Tách riêng OOD C và C++**: Đánh giá riêng biệt trên 2 ngôn ngữ để kiểm tra thiên
     lệch ngôn ngữ (language bias) — mô hình train trên C++ có phát hiện AI trên mã C không?
  3. **Tính TNR (True Negative Rate / Specificity)**: Đánh giá riêng trên nhóm 500 mẫu Human
     OOD — chỉ số này đo khả năng thích nghi miền (Domain Adaptation): tỷ lệ mã Human ngoại
     miền được phân loại đúng là Human (không bị báo sai là AI).
  4. **So sánh LightGBM Cũ vs Mới**: Script hỗ trợ load cả model cũ (44 đặc trưng) và mới
     (32 đặc trưng, retrained) để so sánh trực tiếp hiệu quả của Domain Adaptation.
  5. **Đọc file .cpp thô**: Ngoài JSONL, script cũng hỗ trợ đọc trực tiếp từ thư mục
     test_set_OOD/CPP/ (các file .cpp riêng lẻ, phân theo thư mục con theo tên LLM).

  Script sử dụng concurrent.futures (đa luồng) để tăng tốc trích xuất đặc trưng trên
  tập OOD lớn (40,000 files).

- **Dữ liệu**: test_set_OOD/CPP/ (files .cpp thô) hoặc ood_cpp_test.jsonl
- **Đầu ra**: ood_evaluation_report.json + console output chi tiết


### run_chapter4_experiments.py
- **Loại**: Python Script (68 KB, ~1500 dòng)
- **Nguồn gốc**: Thư mục `tools/`
- **Vai trò**: MÃ NGUỒN CHẠY THỰC NGHIỆM CỦA LUẬN VĂN (BẢN CŨ — alpha=0.48)
- **Mô tả chi tiết**:
  Script tự động hóa (automation) toàn bộ 5 thí nghiệm cốt lõi mô tả trong Chương 4 của
  luận văn tốt nghiệp. Mục đích: chạy 1 lệnh duy nhất để tái tạo lại toàn bộ kết quả.

  **Thực nghiệm 1 — So sánh mô hình ML (Bảng 4.1)**:
  Chạy inference 4 mô hình ML trên 3 tập test ID, xuất bảng Accuracy/F1 so sánh.

  **Thực nghiệm 2 — Giải thích mô hình bằng SHAP (Hình 14, 15)**:
  Tính SHAP values cho LightGBM, vẽ biểu đồ Bar (global importance) và Beeswarm (per-sample
  impact direction). Giải thích tại sao mô hình phân loại AI/Human dựa trên đặc trưng nào.

  **Thực nghiệm 3 — Đánh giá GraphCodeBERT (Bảng 4.2, 4.3)**:
  Load predictions từ GraphCodeBERT 1-Fold, tính Accuracy, F1 trên 3 tập test ID.

  **Thực nghiệm 4 — Mô hình Hybrid (Bảng 4.4, Hình 12)**:
  Grid Search tìm alpha tối ưu trên unified_test: P_hybrid = alpha * P_BERT + (1-alpha) * P_LGBM.
  Kết quả bản cũ: alpha=0.48, đỉnh hiệu năng 93.80% trên unified_test.

  **Thực nghiệm 5 — Ngoại miền OOD (Bảng 4.5-4.7)**:
  Đánh giá Hybrid, ML, DL trên tập OOD 12,599 files (bản cũ).

  **LƯU Ý QUAN TRỌNG**: Script này chứa kết quả thực nghiệm CŨ (alpha=0.48, OOD 12,599 files).
  Bản thực nghiệm MỚI (bài báo khoa học) sử dụng run_ensemble_hybrid_experiment.py tại
  thư mục 3. Deep_Learning_and_Hybrid/ với alpha=0.50 và OOD 39,999 files.

- **Dữ liệu**: Tất cả 5 tập JSONL từ data_splits/
- **Đầu ra**: Bảng kết quả console + biểu đồ PNG


### run_sanity_check_ensemble.py
- **Loại**: Python Script (6 KB, 139 dòng)
- **Nguồn gốc**: Thư mục `tools/`
- **Vai trò**: Kiểm tra tính tương thích trước khi chạy thực nghiệm Ensemble
- **Mô tả chi tiết**:
  Script kiểm tra "sức khỏe" (sanity check) của hệ thống 10-fold Ensemble trước khi chạy
  thực nghiệm lớn trên 40,000 files OOD (tốn hàng giờ GPU). Script kiểm tra:

  1. **Load model**: Thử load tất cả 10 model GraphCodeBERT (mỗi fold 1 model) từ thư mục
     DL models/. Kiểm tra mỗi model có đúng kiến trúc AutoModelForSequenceClassification,
     có output 2 logits (AI/Human), và load thành công trên thiết bị (GPU/CPU).
  2. **Inference thử**: Chạy prediction trên 5 đoạn mã nguồn C++ mẫu (hardcoded) bằng cả
     10 model. Kiểm tra: xác suất output nằm trong [0, 1], tổng 2 class = 1.0.
  3. **Tính Ensemble**: Tính xác suất trung bình từ 10 fold, so sánh với từng fold riêng lẻ.
  4. **Kiểm tra sliding window**: Thử tokenize đoạn mã dài (>512 tokens) bằng kỹ thuật
     sliding window (stride=256), kiểm tra chunking hoạt động đúng.

  Nếu tất cả kiểm tra pass → script in "[PASS] All sanity checks passed" và an toàn chạy
  run_ensemble_hybrid_experiment.py.

- **Yêu cầu**: torch, transformers (Hugging Face), GPU (khuyến nghị)
- **Đầu ra**: Console log pass/fail cho từng kiểm tra


### visualize_pipeline.py
- **Loại**: Python Script (5 KB, ~120 dòng)
- **Nguồn gốc**: Thư mục `retrain ML models/`
- **Vai trò**: Tạo biểu đồ trực quan phục vụ chèn vào luận văn (Chương 3, Chương 4)
- **Mô tả chi tiết**:
  Script tự động vẽ các biểu đồ trực quan hóa phục vụ luận văn:

  1. **Sơ đồ khối Pipeline lọc đặc trưng 3 tầng**: Vẽ block diagram thể hiện luồng xử lý:
     [44 đặc trưng gốc] → Correlation Filter (|r|>0.85) → SelectKBest (MI, k=33)
     → Lasso L1 → [32 đặc trưng tối ưu]. Sử dụng matplotlib với patches và arrows.
     → Xuất: feature_selection_pipeline.png

  2. **Trang HTML tương tác**: Sử dụng Plotly để tạo trang web hiển thị ma trận tương quan
     (correlation matrix heatmap) giữa các đặc trưng trước và sau lọc, scatter plot Mutual
     Information score, và danh sách đặc trưng bị loại tại mỗi tầng.
     → Xuất: feature_selection_visualization.html

  Các file đầu ra được lưu tại 4. Reports_and_Visualizations/.

- **Đầu ra**: feature_selection_pipeline.png, feature_selection_visualization.html


================================================================================

## Thứ tự chạy các script (Workflow)

```
1. create_test_sets_improved.py     → Tạo data_splits/ (train/test/OOD)
2. retrain_improved.py              → Huấn luyện 4 mô hình ML + SHAP
3. generate_baselines_improved.py   → Tạo baselines.json cho web UI
4. inference_improved.py            → Đánh giá ML trên 5 tập test
5. inference_ood.py                 → Phân tích OOD chi tiết theo từng LLM
6. run_chapter4_experiments.py      → Chạy toàn bộ thực nghiệm Chương 4 (bản cũ)
7. visualize_pipeline.py            → Vẽ biểu đồ cho luận văn
```
