# Experiment Report: Optimal Feature Selection k Tuning

Thử nghiệm này chạy trên tập dữ liệu hoàn chỉnh `train_data_with_oop.jsonl` (8,880 records), đánh giá hiệu năng phân loại khi thay đổi tham số $k$ trong tầng lọc `SelectKBest`.

---

## 📊 Kết quả tổng quát

### 1. Phương pháp 1: Grid Search quét tìm k tối ưu
Dưới đây là bảng đánh giá hiệu năng các mô hình theo từng giá trị $k$ của SelectKBest:

| k | Số features sau Lasso L1 | LightGBM Val F1 | XGBoost Val F1 | RandomForest Val F1 | LGBM Overfit (Train-Val F1) |
|---|---|---|---|---|---|
| 5 | 5 | 0.8694 | 0.8754 | 0.8476 | 0.0473 |
| 7 | 7 | 0.9044 | 0.9080 | 0.8837 | 0.0484 |
| 9 | 9 | 0.9074 | 0.9081 | 0.8813 | 0.0566 |
| 11 | 11 | 0.9106 | 0.9127 | 0.8848 | 0.0537 |
| 13 | 13 | 0.9126 | 0.9074 | 0.8832 | 0.0638 |
| 15 | 15 | 0.9207 | 0.9226 | 0.8854 | 0.0631 |
| 17 | 17 | 0.9166 | 0.9193 | 0.8854 | 0.0659 |
| 19 | 19 | 0.9255 | 0.9277 | 0.8938 | 0.0631 |
| 20 | 20 | 0.9266 | 0.9231 | 0.8936 | 0.0599 |
| 21 | 21 | 0.9225 | 0.9258 | 0.8943 | 0.0650 |
| 23 | 23 | 0.9334 | 0.9317 | 0.8982 | 0.0589 |
| 25 | 25 | 0.9277 | 0.9283 | 0.8939 | 0.0656 |
| 27 | 25 | 0.9385 | 0.9350 | 0.9008 | 0.0529 |
| 29 | 28 | 0.9372 | 0.9339 | 0.9012 | 0.0572 |
| 31 | 30 | 0.9456 | 0.9461 | 0.9005 | 0.0514 |
| 33 | 32 | 0.9458 | 0.9450 | 0.9085 | 0.0516 |
| 35 | 33 | 0.9457 | 0.9424 | 0.9079 | 0.0519 |
| 37 | 35 | 0.9430 | 0.9462 | 0.9086 | 0.0552 |
| 39 | 36 | 0.9444 | 0.9445 | 0.9094 | 0.0529 |
| 40 | 36 | 0.9444 | 0.9445 | 0.9094 | 0.0529 |

### 2. Phương pháp 2: RFECV (Recursive Feature Elimination với Cross-Validation)
- **Thuật toán cơ sở**: LightGBM Classifier (150 trees, class_weight='balanced')
- **Số Fold kiểm thử chéo**: 5-Fold Stratified CV
- **Số lượng đặc trưng tối ưu tìm được**: **29 đặc trưng** (trên tổng số 40 đặc trưng sau bộ lọc tương quan)
- **Hiệu năng trên tập Validation (F1-score)**: **0.9440**
- **Hiệu năng trên tập Test kín (F1-score)**: **0.9548**

Các đặc trưng được RFECV giữ lại:
`['comment_ratio', 'empty_line_ratio', 'avg_line_length', 'max_line_length', 'tab_vs_space_ratio', 'trailing_space_ratio', 'brace_style_consistency', 'avg_identifier_length', 'identifier_length_variance', 'single_char_var_ratio', 'unique_identifier_ratio', 'keyword_to_identifier_ratio', 'halstead_volume', 'halstead_difficulty', 'maintainability_index', 'code_to_comment_ratio', 'max_nesting_depth', 'total_includes', 'macro_count', 'modern_cpp_ratio', 'const_usage_ratio', 'shannon_entropy', 'class_count', 'struct_count', 'access_specifier_ratio', 'using_std_ratio', 'raw_pointer_ratio', 'getter_setter_ratio', 'std_prefix_ratio']`

---

## 💡 Đánh giá & Khuyến nghị

1. **Về giá trị k mặc định (k = 20)**:
   - Hiệu năng của LightGBM ở k = 20 có F1-score là **0.9266**.
   - Số đặc trưng thực tế đi qua Lasso L1 là **20**.

2. **Về giá trị k tối ưu nhất qua Grid Search**:
   - Giá trị $k$ tối ưu nhất cho LightGBM là **k = 33** với Val F1 đạt **0.9458** (Đặc trưng sau Lasso: **32**).
   - Điểm tối ưu này giúp tăng thêm hiệu năng so với mặc định mà không làm tăng quá nhiều nguy cơ overfit.

3. **Về phương pháp RFECV**:
   - RFECV tìm ra tập **29** đặc trưng tối ưu đạt Val F1 là **0.9440**.
   - RFECV mang lại độ chính xác cao nhưng đòi hỏi thay đổi kiến trúc code (loại bỏ SelectKBest và Lasso L1, thay bằng bộ lọc RFECV đã được huấn luyện sẵn hoặc chạy RFECV trực tiếp trong pipeline huấn luyện).

### 🏆 Đề xuất hành động:
- Nếu muốn giữ nguyên pipeline 3 tầng ổn định hiện tại, hãy cập nhật **k = 33** (hoặc con số được chọn dựa trên đồ thị uốn).
- Nếu muốn tối ưu sâu hơn nữa và tự động hóa 100%, chúng ta có thể chuyển sang dùng RFECV lưu checkpoint `.pkl`.
