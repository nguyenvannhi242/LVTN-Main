# DỮ LIỆU THỰC NGHIỆM (Datasets)

Thư mục này chứa toàn bộ các tập dữ liệu đã được sử dụng trong các thực nghiệm của luận văn tốt nghiệp
và bài báo khoa học. Tất cả file dữ liệu đều ở định dạng JSONL (JSON Lines — mỗi dòng là một object JSON
độc lập).

Nguồn gốc: sao chép từ thư mục data_splits/ của dự án chính.

Tổng dung lượng: ~242 MB (8 files)


================================================================================

## CẤU TRÚC THƯ MỤC

```
5. Datasets/
├── train_set/                          # Dữ liệu huấn luyện
│   ├── train_data_10_model_AI.jsonl    # 15.88 MB — Dữ liệu gốc Basic C++
│   ├── OOP_cpp_ai_detection_dataset.jsonl  # 11.36 MB — Dữ liệu OOP C++
│   └── train_data_with_oop.jsonl       # 27.07 MB — Dữ liệu gộp (chính thức)
│
├── test_set_ID/                        # Tập kiểm thử nội miền (In-Distribution)
│   ├── basic_test.jsonl                # 1.85 MB — 500 mẫu Basic C++
│   └── oop_test.jsonl                  # 1.72 MB — 500 mẫu OOP C++
│
├── test_set_unified/                   # Tập kiểm thử hợp nhất
│   └── unified_test.jsonl              # 3.57 MB — 1000 mẫu (Basic + OOP)
│
├── test_set_OOD/                       # Tập kiểm thử ngoại miền (Out-of-Distribution)
│   ├── ood_c_test.jsonl                # 86.98 MB — 20,000 files OOD ngôn ngữ C
│   └── ood_cpp_test.jsonl              # 93.83 MB — 19,999 files OOD ngôn ngữ C++
│
└── README.md                           # File này
```


================================================================================

## ĐỊNH DẠNG DỮ LIỆU (JSONL Schema)

Mỗi dòng trong file JSONL là một JSON object có cấu trúc:

```json
{
  "code": "<mã nguồn C/C++ nguyên bản>",
  "label": "AI" hoặc "HUMAN",
  "model": "<tên mô hình AI sinh mã>" hoặc "human",
  "language": "c" hoặc "cpp",
  ...
}
```

Trong đó:
- "code": Chuỗi mã nguồn C/C++ hoàn chỉnh (bao gồm cả #include, hàm main, v.v.)
- "label": Nhãn phân loại nhị phân — "AI" nếu mã do AI sinh, "HUMAN" nếu do người viết
- "model": Tên cụ thể của mô hình AI đã sinh ra mã (chỉ có khi label="AI")
- "language": Ngôn ngữ lập trình (c hoặc cpp)

Lưu ý: Một số file cũ hơn (train_data_10_model_AI.jsonl) có thể có schema hơi khác
(thiếu trường "language"), nhưng luôn có "code" và "label".


================================================================================

## MÔ TẢ CHI TIẾT TỪNG TẬP DỮ LIỆU


### 1. train_set/ — Dữ liệu Huấn luyện

#### train_data_10_model_AI.jsonl (15.88 MB)
  Tập dữ liệu gốc dùng để huấn luyện mô hình phát hiện mã AI, chỉ bao gồm mã nguồn C++ cơ bản
  (Basic C++ — bài tập thuật toán, cấu trúc dữ liệu cơ bản, nhập xuất đơn giản).

  Nguồn dữ liệu mã AI: Mã C++ được sinh tự động bởi 10 mô hình ngôn ngữ lớn (Large Language Models)
  hàng đầu thế giới. Mỗi mô hình được yêu cầu giải lại các bài tập lập trình tương tự với bài của
  sinh viên thực.

  Nguồn dữ liệu mã Human: Bài tập lập trình C++ thực tế của sinh viên, thu thập từ các hệ thống
  chấm bài trực tuyến (Online Judge) và các khóa học đại học.

  Sử dụng bởi: retrain.py, retrain_improved.py, find_optimal_k.py

#### OOP_cpp_ai_detection_dataset.jsonl (11.36 MB)
  Tập dữ liệu bổ sung chứa mã nguồn C++ hướng đối tượng (Object-Oriented Programming — OOP). Dữ liệu
  bao gồm các bài tập nâng cao hơn với cấu trúc class, kế thừa (inheritance), đa hình (polymorphism),
  đóng gói (encapsulation), và các design pattern.

  Mục đích: Mở rộng khả năng tổng quát hóa của mô hình từ phong cách Basic C++ sang phong cách OOP,
  vì mã OOP có đặc trưng cấu trúc rất khác so với mã thuật toán cơ bản (VD: số class, độ sâu kế thừa,
  số method, tỷ lệ đóng gói — đây là các đặc trưng mà bộ CppFeatureExtractorV8 trích xuất).

  Sử dụng bởi: retrain.py (lấy mẫu 300 AI + 300 Human), retrain_improved.py

#### train_data_with_oop.jsonl (27.07 MB)
  TẬP HUẤN LUYỆN CHÍNH THỨC — kết quả gộp (merge) của train_data_10_model_AI.jsonl và
  OOP_cpp_ai_detection_dataset.jsonl. Đây là tập dữ liệu được sử dụng trực tiếp trong pipeline
  huấn luyện chính thức (retrain_improved.py) và pipeline lọc đặc trưng (find_optimal_k.py).

  Tổng số bản ghi: ~8,880 records (cân bằng giữa AI và Human).

  Sử dụng bởi: retrain_improved.py (chính), find_optimal_k.py (chính)


### 2. test_set_ID/ — Tập kiểm thử Nội miền (In-Distribution)

Dữ liệu kiểm thử có cùng phân phối (distribution) với dữ liệu huấn luyện. Được tách ra từ cùng
nguồn dữ liệu gốc nhưng KHÔNG trùng lặp với tập train (holdout split).

#### basic_test.jsonl (1.85 MB)
  500 mẫu mã nguồn C++ cơ bản (Basic C++), cân bằng 50/50 giữa AI và Human.
  Đại diện cho phong cách lập trình thuật toán đơn giản: vòng lặp, mảng, chuỗi, đệ quy, sắp xếp.

  Dùng để đánh giá: Hiệu năng cơ bản (baseline performance) của mô hình trên dữ liệu thuần Basic.
  Tương ứng với cột "basic_test" trong Bảng 4.1-4.4 của luận văn.

  Sử dụng bởi: inference_improved.py, run_chapter4_experiments.py, run_ensemble_hybrid_experiment.py

#### oop_test.jsonl (1.72 MB)
  500 mẫu mã nguồn C++ hướng đối tượng (OOP C++), cân bằng 50/50 giữa AI và Human.
  Đại diện cho phong cách lập trình nâng cao: class, method, inheritance, template, operator overloading.

  Dùng để đánh giá: Khả năng tổng quát hóa sang phong cách OOP — kiểm tra xem mô hình huấn luyện
  chủ yếu trên Basic có phân biệt được AI/Human trong mã OOP hay không.
  Tương ứng với cột "oop_test" trong Bảng 4.1-4.4 của luận văn.

  Sử dụng bởi: inference_improved.py, run_chapter4_experiments.py, run_ensemble_hybrid_experiment.py


### 3. test_set_unified/ — Tập kiểm thử Hợp nhất

#### unified_test.jsonl (3.57 MB)
  1,000 mẫu mã nguồn C++ hợp nhất cả Basic (500) và OOP (500), cân bằng AI/Human.
  Đây là tập kiểm thử CHÍNH được sử dụng để:
    - Tìm hệ số lai alpha tối ưu (Grid Search trên unified_test)
    - Báo cáo hiệu năng tổng thể của mô hình Hybrid
    - So sánh ML vs DL vs Hybrid

  Tương ứng với cột "unified_test" trong Bảng 4.1-4.4 và là tập được dùng trong Hình 12 (đồ thị
  Grid Search alpha) của luận văn.

  Sử dụng bởi: inference_improved.py, run_chapter4_experiments.py, run_ensemble_hybrid_experiment.py


### 4. test_set_OOD/ — Tập kiểm thử Ngoại miền (Out-of-Distribution)

Dữ liệu kiểm thử có phân phối HOÀN TOÀN KHÁC so với dữ liệu huấn luyện. Mã nguồn được lấy từ
nguồn dữ liệu độc lập (CodeMirage) với các bài toán, phong cách và mô hình AI sinh mã khác biệt.
Mục đích: Đánh giá khả năng tổng quát hóa thực sự (real-world generalization) của mô hình khi gặp
dữ liệu chưa từng thấy.

#### ood_c_test.jsonl (86.98 MB)
  20,000 files mã nguồn ngôn ngữ C (không phải C++), gồm:
    - 10,000 mẫu AI: Được sinh bởi 10 mô hình LLM khác nhau (mỗi mô hình ~1,000 mẫu):
      GPT-4o-mini, DeepSeek-R1, DeepSeek-V3, Claude-3.5-Haiku, Gemini-2.0-Flash,
      Gemini-2.0-Flash-Thinking-Exp, Gemini-2.0-Pro-Exp, Llama3.3-70B, O3-mini, Qwen2.5-Coder
    - 10,000 mẫu Human: Mã C thực tế viết bởi lập trình viên

  Thách thức ngoại miền: Mô hình được huấn luyện trên mã C++ nhưng phải phát hiện mã AI viết
  bằng ngôn ngữ C — khác biệt cú pháp (không có class, namespace, STL) nhưng có thể giữ lại
  các đặc trưng phong cách (comment, naming, formatting).

  Tương ứng với cột "OOD C" trong Bảng 4.5-4.7 của luận văn và Bảng XIV-XVIII của bài báo.

  Sử dụng bởi: inference_improved.py, inference_ood.py, run_ensemble_hybrid_experiment.py

#### ood_cpp_test.jsonl (93.83 MB)
  19,999 files mã nguồn ngôn ngữ C++, gồm:
    - ~10,000 mẫu AI: Được sinh bởi cùng 10 mô hình LLM như trên, nhưng với các bài toán
      C++ hoàn toàn khác so với tập huấn luyện
    - ~10,000 mẫu Human: Mã C++ thực tế viết bởi lập trình viên (nguồn khác với tập train)

  Thách thức ngoại miền: Tuy cùng ngôn ngữ C++ nhưng phong cách bài toán, cấu trúc code và
  mức độ phức tạp rất khác so với dữ liệu huấn luyện (bài tập sinh viên). Đây là tập OOD
  khó nhất — nơi LightGBM Cũ chỉ đạt 22.53% accuracy trước khi thích nghi miền.

  Tương ứng với cột "OOD C++" trong Bảng 4.5-4.7 của luận văn và Bảng XIV-XVIII của bài báo.

  Sử dụng bởi: inference_improved.py, inference_ood.py, run_ensemble_hybrid_experiment.py


================================================================================

## THỐNG KÊ TỔNG QUAN

| Tập dữ liệu | Số lượng mẫu | Tỷ lệ AI:Human | Ngôn ngữ | Vai trò |
|:---|:---:|:---:|:---:|:---|
| train_data_10_model_AI | ~7,000 | ~50:50 | C++ (Basic) | Huấn luyện (gốc) |
| OOP_cpp_ai_detection_dataset | ~1,880 | ~50:50 | C++ (OOP) | Huấn luyện (bổ sung) |
| train_data_with_oop | ~8,880 | ~50:50 | C++ (Basic+OOP) | Huấn luyện (chính thức) |
| basic_test | 500 | 50:50 | C++ (Basic) | Kiểm thử ID |
| oop_test | 500 | 50:50 | C++ (OOP) | Kiểm thử ID |
| unified_test | 1,000 | 50:50 | C++ (Basic+OOP) | Kiểm thử ID (chính) |
| ood_c_test | 20,000 | 50:50 | C | Kiểm thử OOD |
| ood_cpp_test | 19,999 | ~50:50 | C++ | Kiểm thử OOD |

Tổng: ~51,759 mẫu mã nguồn


================================================================================

## MỐI LIÊN HỆ VỚI CÁC THỰC NGHIỆM

### Thực nghiệm 1 — So sánh mô hình ML (Chương 4.3)
  Dữ liệu sử dụng: train_data_with_oop.jsonl (huấn luyện) + basic_test, oop_test, unified_test (đánh giá)
  Script: retrain_improved.py, inference_improved.py

### Thực nghiệm 2 — Giải thích mô hình bằng SHAP (Chương 4.4)
  Dữ liệu sử dụng: train_data_with_oop.jsonl (tính SHAP trên tập train)
  Script: retrain_improved.py (phần SHAP)

### Thực nghiệm 3 — Đánh giá GraphCodeBERT (Chương 4.5)
  Dữ liệu sử dụng: train_data_with_oop.jsonl (10-fold LOGO cross-validation)
  Script: Basic_of_LVTN_Graphcodebert_tính_micro_&_macro.ipynb

### Thực nghiệm 4 — Mô hình Lai Hybrid (Chương 4.6)
  Dữ liệu sử dụng: unified_test.jsonl (Grid Search alpha), basic_test + oop_test (đánh giá bổ sung)
  Script: run_ensemble_hybrid_experiment.py (mới, alpha=0.50), Hybrid_model.ipynb (cũ, alpha=0.48)

### Thực nghiệm 5 — Ngoại miền & Thích nghi miền (Chương 4.7)
  Dữ liệu sử dụng: ood_c_test.jsonl + ood_cpp_test.jsonl (39,999 files OOD)
  Script: run_ensemble_hybrid_experiment.py, inference_ood.py
