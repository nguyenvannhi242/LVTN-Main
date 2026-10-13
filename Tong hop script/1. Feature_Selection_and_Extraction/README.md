# Trích xuất & Tuyển chọn Đặc trưng (Feature Selection & Extraction)

Thư mục này chứa toàn bộ mã nguồn phục vụ giai đoạn **xây dựng bộ trích xuất đặc trưng tĩnh** từ mã
nguồn C/C++ và **quy trình tuyển chọn đặc trưng 3 tầng** (3-Stage Feature Selection Pipeline) để
giảm số lượng đặc trưng từ 44 gốc xuống còn 32 đặc trưng tối ưu.

Đây là giai đoạn nền tảng, cung cấp đầu vào cho toàn bộ hệ thống phát hiện mã AI:
bộ trích xuất CppFeatureExtractorV8 (được định nghĩa lần đầu trong các notebook ở đây) sau đó được
tích hợp vào pipeline huấn luyện (retrain_improved.py) và pipeline suy luận (inference_improved.py).


## Danh sách file (4 files)

```
1. Feature_Selection_and_Extraction/
├── Extract_Feature_Original.ipynb      # 741 KB — Notebook thử nghiệm ban đầu
├── Extract_Feature_ML.ipynb            # 625 KB — Notebook chuẩn hóa cho ML pipeline
├── extracted_with_markdown.py          # 46 KB  — Script trích xuất từ JSONL/Markdown
└── find_optimal_k.py                   # 16 KB  — Script lọc đặc trưng 3 tầng
```


================================================================================

## Chi tiết từng file


### Extract_Feature_Original.ipynb
- **Loại**: Jupyter Notebook (741 KB)
- **Nguồn gốc**: Thư mục `Extract feature/` của dự án
- **Vai trò trong luận văn**: Giai đoạn nghiên cứu khám phá (exploratory research)
- **Mô tả chi tiết**:
  Notebook gốc ban đầu, thực hiện nghiên cứu thử nghiệm (trial-and-error) trích xuất đặc trưng
  tĩnh từ các bài tập lập trình C++ cơ bản của sinh viên. Notebook bao gồm:

  1. Đọc file mã nguồn C++ thô từ nhiều nguồn (file .cpp đơn lẻ, code block trong Markdown)
  2. Áp dụng kỹ thuật phân tích cú pháp bằng biểu thức chính quy (regex) để nhận diện:
     - Kiểu chú thích (comment style), tỷ lệ dòng trống, kiểu thụt lề (tab vs space)
     - Quy ước đặt tên biến (naming convention), tỷ lệ biến 1 ký tự
     - Từ khóa C++ đặc thù (bits/stdc++.h, ios::sync_with_stdio, macro #define)
  3. Sử dụng thư viện `lizard` để tính độ phức tạp vòng lặp (Cyclomatic Complexity)
  4. Tính toán các chỉ số lý thuyết thông tin: Shannon Entropy, Bigram Entropy
  5. Tính toán bộ chỉ số Halstead (Volume, Difficulty, Effort, Estimated Bugs)
  6. Thử nghiệm các đặc trưng OOP: class_count, inheritance_depth, method_count, encapsulation_ratio

  Notebook này mang tính thăm dò — nhiều đặc trưng được thử rồi loại bỏ. Kết quả cuối cùng
  được đúc kết thành class `CppFeatureExtractorV8` (phiên bản chính thức).

- **Đầu vào**: File mã nguồn C++ (.cpp) hoặc JSONL chứa trường "code"
- **Đầu ra**: DataFrame Pandas chứa các cột đặc trưng số (numerical features)


### Extract_Feature_ML.ipynb
- **Loại**: Jupyter Notebook (625 KB)
- **Nguồn gốc**: Thư mục `Extract feature/` của dự án
- **Vai trò trong luận văn**: Chuẩn hóa bộ trích xuất cho pipeline ML
- **Mô tả chi tiết**:
  Phiên bản chuẩn hóa và tối ưu hóa của Extract_Feature_Original.ipynb, được thiết kế để tích hợp
  trực tiếp vào pipeline học máy (Machine Learning pipeline). Các thay đổi chính so với bản gốc:

  1. **Chuẩn hóa schema đầu ra**: Đảm bảo DataFrame có schema cố định (tên cột, kiểu dữ liệu, thứ
     tự) nhất quán giữa các lần chạy, phục vụ cho việc StandardScaler và model inference.
  2. **Xử lý trường hợp biên (edge cases)**: Xử lý đúng khi mã nguồn quá ngắn (< 3 dòng), file
     rỗng, hoặc mã nguồn không hợp lệ (không compile được nhưng vẫn cần trích xuất đặc trưng).
  3. **Bổ sung đặc trưng OOP**: Tích hợp 4 đặc trưng OOP mới:
     - `class_count`: Số lượng class/struct được khai báo
     - `inheritance_depth`: Độ sâu cây kế thừa (tính bằng regex đếm `:` sau tên class)
     - `method_count`: Số phương thức (method) trong các class
     - `encapsulation_ratio`: Tỷ lệ trường private/protected so với tổng số trường
  4. **Tinh chỉnh cách tính Complexity metrics**: Sử dụng `lizard` với cấu hình chính xác hơn
     cho ngôn ngữ C++ (xử lý template, operator overloading).

- **Đầu vào**: File JSONL chứa trường "code" và "label"
- **Đầu ra**: DataFrame Pandas đã chuẩn hóa, sẵn sàng cho huấn luyện mô hình


### extracted_with_markdown.py
- **Loại**: Python Script (46 KB, ~1074 dòng)
- **Nguồn gốc**: Thư mục `Extract feature/` (được chuyển đổi từ notebook Colab)
- **Vai trò trong luận văn**: Trích xuất đặc trưng hàng loạt từ dữ liệu JSONL
- **Mô tả chi tiết**:
  Script toàn diện kết hợp cả hai chức năng: (1) định nghĩa class CppFeatureExtractorV8 hoàn chỉnh
  và (2) pipeline trích xuất + huấn luyện sơ bộ. Script được convert từ notebook Colab nên
  giữ nguyên cấu trúc cell.

  **Phần 1 — Bộ trích xuất CppFeatureExtractorV8**:
  Định nghĩa đầy đủ class CppFeatureExtractorV8 với 44 đặc trưng gốc (trước khi lọc), bao gồm:
  - 7 đặc trưng Layout & Formatting (comment_ratio, empty_line_ratio, avg_line_length, ...)
  - 5 đặc trưng Naming Conventions (avg_identifier_length, single_char_var_ratio, ...)
  - 5 đặc trưng Structural Complexity (avg_cyclomatic_complexity, max_nesting_depth, ...)
  - 6 đặc trưng Halstead Metrics (halstead_volume, halstead_difficulty, halstead_effort, ...)
  - 5 đặc trưng Habits & Idioms (total_includes, macro_count, has_bits_stdc, ...)
  - 5 đặc trưng Information Theory (shannon_entropy, bigram_entropy, whitespace_entropy, ...)
  - 4 đặc trưng OOP Metrics (class_count, inheritance_depth, method_count, encapsulation_ratio)
  - Các đặc trưng khác: maintainability_index, modern_cpp_score, const_usage_ratio, ...

  **Phần 2 — Pipeline trích xuất từ JSONL**:
  Đọc file JSONL (train_data_10_model_AI.jsonl), trích xuất đặc trưng cho từng mẫu, xây dựng
  DataFrame và thực hiện huấn luyện thử nghiệm với LightGBM, XGBoost, CatBoost, Random Forest.

  **Phần 3 — Lọc đặc trưng sơ bộ**:
  Thử nghiệm Correlation Filter và SelectKBest để khám phá mối tương quan giữa các đặc trưng.

- **Thư viện yêu cầu**: lizard, xgboost, lightgbm, catboost, shap, scikit-learn, pandas, tqdm
- **Đầu vào**: File JSONL chứa trường "code" (hỗ trợ cả Markdown với code blocks ```cpp)
- **Đầu ra**: DataFrame đặc trưng, mô hình thử nghiệm, biểu đồ SHAP sơ bộ


### find_optimal_k.py
- **Loại**: Python Script (16 KB, 351 dòng)
- **Nguồn gốc**: Thư mục `retrain ML models/`
- **Vai trò trong luận văn**: Thực nghiệm tuyển chọn đặc trưng (Chương 3 — Phương pháp)
- **Mô tả chi tiết**:
  Script cốt lõi thực thi Quy trình lọc đặc trưng 3 tầng (3-Stage Feature Selection Pipeline) —
  một trong những đóng góp phương pháp luận chính của luận văn. Chi tiết từng tầng:

  **Tầng 1 — Correlation Filter (Lọc tương quan Pearson)**:
  Tính ma trận tương quan Pearson giữa tất cả 44 đặc trưng gốc. Nếu cặp đặc trưng nào có hệ số
  tương quan |r| > 0.85, giữ lại đặc trưng có Mutual Information cao hơn (tức có khả năng phân
  biệt AI/Human tốt hơn) và loại bỏ đặc trưng dư thừa còn lại. Mục đích: giảm đa cộng tuyến
  (multicollinearity) để tránh nhiễu cho mô hình.

  **Tầng 2 — SelectKBest với Mutual Information**:
  Sử dụng sklearn.feature_selection.mutual_info_classif để đánh giá mức độ phụ thuộc phi tuyến
  (non-linear dependency) giữa từng đặc trưng và nhãn AI/Human. Chọn ra top k đặc trưng có
  Mutual Information cao nhất. Script quét k từ 5 đến 40 (bước 2) để tìm điểm uốn tối ưu.

  **Tầng 3 — Lasso L1 Regularization**:
  Áp dụng Logistic Regression với penalty='l1' (Lasso) lên tập k đặc trưng đã chọn. Thuật toán
  Lasso tự động ép trọng số (coefficient) của các đặc trưng dư thừa về 0, giữ lại chỉ những
  đặc trưng thực sự đóng góp vào khả năng phân biệt. Ví dụ: với k=33, Lasso loại 1 đặc trưng
  và giữ lại 32 đặc trưng cuối cùng.

  **Đánh giá tại mỗi giá trị k**:
  Tại mỗi giá trị k, script huấn luyện 4 mô hình (LightGBM, XGBoost, Random Forest, Logistic
  Regression) bằng Stratified K-Fold CV, ghi lại Accuracy và F1-score. Kết quả được tổng hợp
  thành bảng và đồ thị để xác định k tối ưu.

  **Phương pháp bổ sung — RFECV**:
  Ngoài Grid Search, script cũng chạy Recursive Feature Elimination with Cross-Validation (RFECV)
  với LightGBM làm estimator base, cho kết quả tham chiếu chéo (29 đặc trưng tối ưu, F1=0.9548).

- **Dữ liệu đầu vào**: data_splits/train_set/train_data_with_oop.jsonl (~8,880 records)
- **Đầu ra**:
  - optimal_k_report.md (báo cáo bảng số liệu) → lưu tại 4. Reports_and_Visualizations/
  - optimal_k_search.png (đồ thị F1 vs k) → lưu tại 4. Reports_and_Visualizations/
- **Kết luận**: k=33 (sau Lasso → 32 đặc trưng) là cấu hình chuẩn cho hệ thống


================================================================================

## Mối liên hệ giữa các file

```
Extract_Feature_Original.ipynb    (thử nghiệm ban đầu)
        │
        ▼
Extract_Feature_ML.ipynb          (chuẩn hóa + bổ sung OOP)
        │
        ▼
extracted_with_markdown.py        (đóng gói class CppFeatureExtractorV8 + pipeline)
        │
        ▼
find_optimal_k.py                 (lọc 3 tầng: 44 → 32 đặc trưng)
        │
        ▼
  [Output] → retrain_improved.py (thư mục 2.) sử dụng 32 đặc trưng để huấn luyện
```
