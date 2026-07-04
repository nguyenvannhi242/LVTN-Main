# TỔNG HỢP MÃ NGUỒN THỰC NGHIỆM VÀ ĐÁNH GIÁ (LUẬN VĂN TỐT NGHIỆP)

Thư mục này tổng hợp toàn bộ các mã nguồn thực nghiệm, mã huấn luyện (train), suy luận (inference), tìm tham số tối ưu và các báo cáo kết quả thực nghiệm phục vụ trực tiếp cho việc viết luận văn tốt nghiệp và bài báo khoa học.

Các mã nguồn này tách biệt với mã nguồn chạy ứng dụng web chính thức (nằm trong thư mục src/) nhằm đảm bảo tính gọn nhẹ và dễ theo dõi. Mỗi file đều có nguồn gốc từ các thư mục làm việc khác nhau trong dự án (retrain ML models/, Extract feature/, tools/, testing ML models/) và đã được sao chép tập trung về đây.

Cấu trúc thư mục được chia làm 5 phần chính:
1. Trích xuất & Tuyển chọn Đặc trưng (Feature Selection & Extraction)
2. Huấn luyện & Suy luận Mô hình Học máy (ML Training & Inference)
3. Mô hình Học sâu & Mô hình Lai (Deep Learning & Hybrid)
4. Báo cáo Thực nghiệm & Trực quan hóa (Reports & Visualizations)
5. Dữ liệu (Datasets) — các tập dữ liệu đã sử dụng trong thực nghiệm

================================================================================

## 1. TRÍCH XUẤT & TUYỂN CHỌN ĐẶC TRƯNG (1. Feature_Selection_and_Extraction/)

Thư mục này lưu trữ mã nguồn dùng để phát triển bộ trích xuất đặc trưng tĩnh từ mã nguồn C++ và quy trình lọc đặc trưng 3 tầng để giảm số lượng đặc trưng từ 44 xuống còn 32 đặc trưng tối ưu.


### Extract_Feature_Original.ipynb
  - Loại: Jupyter Notebook
  - Nguồn gốc: Extract feature/
  - Mô tả: Notebook gốc ban đầu, thực hiện nghiên cứu thử nghiệm (exploratory) trích xuất đặc trưng tĩnh
    từ các bài tập lập trình C++ cơ bản của sinh viên. Notebook này chứa các đoạn mã thử-sai (trial-and-error)
    trong giai đoạn đầu xây dựng hệ thống, bao gồm: đọc file mã nguồn C++ thô, áp dụng các kỹ thuật phân
    tích cú pháp (parsing) bằng regex và thư viện tree-sitter, tính toán các chỉ số như độ phức tạp vòng lặp
    (Cyclomatic Complexity), entropy Shannon, chỉ số Halstead, và các thống kê về phong cách viết mã (coding
    style statistics). Đây là tiền thân trước khi được đóng gói thành class CppFeatureExtractorV8.

### Extract_Feature_ML.ipynb
  - Loại: Jupyter Notebook
  - Nguồn gốc: Extract feature/
  - Mô tả: Phiên bản chuẩn hóa của Extract_Feature_Original.ipynb, được tối ưu để phục vụ trực tiếp cho
    pipeline học máy (Machine Learning pipeline). Notebook này tinh chỉnh lại cách tính toán các chỉ số
    độ phức tạp mã nguồn (Complexity metrics) để đảm bảo tính nhất quán khi dùng cho huấn luyện mô hình.
    Các thay đổi chính so với bản gốc: (1) chuẩn hóa đầu ra thành DataFrame Pandas với schema cố định,
    (2) xử lý các trường hợp biên (edge cases) khi mã nguồn quá ngắn hoặc không hợp lệ, (3) bổ sung
    các đặc trưng phong cách OOP (class_count, inheritance_depth, method_count, encapsulation_ratio).

### extracted_with_markdown.py
  - Loại: Python Script
  - Nguồn gốc: Extract feature/
  - Mô tả: Script tiện ích hỗ trợ trích xuất đặc trưng trực tiếp từ các đoạn mã C++ nằm lồng bên trong
    các file tài liệu Markdown (.md) hoặc các file JSONL. Script tự động phát hiện các khối code C++
    (code blocks) trong file Markdown thông qua cú pháp "```cpp ... ```", tách chúng ra, và chạy bộ trích
    xuất CppFeatureExtractorV8 trên từng đoạn. Kết quả được xuất ra dưới dạng CSV hoặc DataFrame. Script
    này đặc biệt hữu ích khi cần xử lý hàng loạt dữ liệu đã được chú thích (annotated data) từ các
    dataset ngoại miền được lưu trữ dưới dạng JSONL.

### find_optimal_k.py
  - Loại: Python Script (~16KB, ~300 dòng)
  - Nguồn gốc: retrain ML models/
  - Mô tả: Script cốt lõi thực thi Quy trình lọc đặc trưng 3 tầng (3-stage Feature Selection Pipeline),
    đây là một trong những đóng góp phương pháp luận quan trọng của luận văn. Chi tiết 3 tầng:

    Tầng 1 — Correlation Filter (Lọc tương quan Pearson):
      Tính ma trận tương quan Pearson giữa tất cả 44 đặc trưng gốc. Nếu cặp đặc trưng nào có hệ số
      tương quan |r| > 0.85 (ngưỡng hardcode), giữ lại đặc trưng có Information Gain cao hơn và loại bỏ
      đặc trưng còn lại. Mục đích: giảm đa cộng tuyến (multicollinearity) trước khi huấn luyện.

    Tầng 2 — SelectKBest với Mutual Information:
      Sử dụng sklearn.feature_selection.mutual_info_classif để đánh giá mức độ phụ thuộc phi tuyến giữa
      từng đặc trưng và nhãn (AI/Human). Chọn ra top k đặc trưng có Mutual Information cao nhất. Script
      thực hiện quét giá trị k từ 5 đến 40 (bước 1) để tìm điểm uốn tối ưu (K-elbow).

    Tầng 3 — Lasso L1 Regularization:
      Áp dụng Logistic Regression với penalty L1 (Lasso) lên tập k đặc trưng đã chọn. Lasso tự động
      ép các trọng số đặc trưng dư thừa về 0, giữ lại chỉ những đặc trưng thực sự có đóng góp phân
      biệt. Kết quả cuối cùng: từ k=33 đầu vào, Lasso L1 loại 1 đặc trưng và giữ lại 32 đặc trưng
      tối ưu.

    Script cũng tự động huấn luyện 4 mô hình ML (LightGBM, XGBoost, Random Forest, Logistic Regression)
    tại mỗi giá trị k, ghi lại Accuracy và F1-score để vẽ đồ thị so sánh, và xuất ra báo cáo tại
    optimal_k_report.md.

### quy_trinh_tuyen_chon_dac_trung.txt
  - Loại: File ghi chú văn bản thuần (plain text)
  - Nguồn gốc: Tạo trong quá trình nghiên cứu
  - Mô tả: File ghi chú tóm tắt các yêu cầu thiết kế, kết quả trung gian và định hướng viết báo cáo
    cho quá trình lọc đặc trưng. Nội dung bao gồm: (1) ghi chú lý do chọn ngưỡng tương quan r=0.85,
    (2) giải thích ý nghĩa hệ số lai tối ưu alpha=0.5 (trọng số ngang bằng giữa nhánh ML và DL),
    (3) định nghĩa chỉ số True Negative Rate (TNR) = Specificity trong đánh giá thích nghi miền (Domain
    Adaptation) — tức tỷ lệ mã nguồn Human bị phân loại đúng là Human trên tập dữ liệu ngoại miền OOD,
    (4) danh sách 32 đặc trưng được giữ lại sau pipeline lọc.

================================================================================

## 2. HUẤN LUYỆN & SUY LUẬN MÔ HÌNH HỌC MÁY (2. ML_Training_and_Inference/)

Tổng hợp mã nguồn xây dựng tập dữ liệu hỗn hợp (Basic C++ + OOP C++), huấn luyện các mô hình Machine
Learning cổ điển (LightGBM, XGBoost, Random Forest, Logistic Regression) và đánh giá trên các tập dữ liệu
nội miền (In-Distribution) và ngoại miền (Out-of-Distribution).


### retrain.py
  - Loại: Python Script (~29KB)
  - Nguồn gốc: retrain ML models/
  - Mô tả: Phiên bản đầu tiên (v1) của quy trình huấn luyện lại mô hình ML. Script thực hiện: (1) đọc
    tập dữ liệu gốc train_data_10_model_AI.jsonl (dữ liệu Basic C++ từ 10 mô hình AI), (2) đọc thêm
    tập OOP_cpp_ai_detection_dataset.jsonl (dữ liệu OOP C++), (3) lấy ngẫu nhiên 300 mẫu AI và 300
    mẫu Human từ dataset OOP để gộp vào tập huấn luyện (stratified sampling), (4) trích xuất toàn bộ
    44 đặc trưng tĩnh ban đầu (chưa qua pipeline lọc 3 tầng), (5) huấn luyện các mô hình cơ bản
    (LightGBM, XGBoost, Random Forest) với cấu hình hyperparameter mặc định, (6) lưu model dưới dạng
    .pkl. Đây là phiên bản sơ khai, sau này được thay thế bởi retrain_improved.py.

### retrain_improved.py
  - Loại: Python Script (~25KB)
  - Nguồn gốc: retrain ML models/
  - Mô tả: Script huấn luyện CHÍNH THỨC (production-ready) — sản phẩm cải tiến của retrain.py. Các
    điểm nâng cấp quan trọng:

    (1) Tích hợp class CppFeatureExtractorV8 trực tiếp (32 đặc trưng tĩnh sau khi đã qua pipeline lọc
        3 tầng), thay vì dùng 44 đặc trưng thô.
    (2) Huấn luyện 4 mô hình song song: LightGBM (chính), XGBoost, Random Forest, Logistic Regression.
    (3) Sử dụng kiểm thử chéo phân tầng (Stratified K-Fold Cross-Validation) với k=5 để đánh giá
        hiệu năng một cách khách quan hơn.
    (4) Tự động tính toán và in ra các chỉ số: Accuracy, F1-score (Macro & Micro), Precision, Recall,
        Confusion Matrix cho từng fold.
    (5) Tích hợp SHAP (SHapley Additive exPlanations) để tính giá trị SHAP cho từng đặc trưng, phục vụ
        cho thực nghiệm XAI (Explainable AI) trong Chương 4.
    (6) Lưu trữ các model đã train dưới định dạng .joblib (hiệu quả hơn .pkl cho scikit-learn).
    (7) Tự động xuất file scaler.pkl (StandardScaler đã fit) và final_features.pkl (danh sách 32 đặc
        trưng) để sử dụng trong inference.

### create_test_sets_improved.py
  - Loại: Python Script (~17KB)
  - Nguồn gốc: retrain ML models/
  - Mô tả: Script chuẩn bị và phân tách tập dữ liệu kiểm thử (test set). Script đọc dữ liệu từ các
    nguồn khác nhau (Basic C++, OOP C++, OOD) và tạo ra cấu trúc thư mục data_splits/ với các tập
    kiểm thử riêng biệt: basic_test.jsonl (500 mẫu Basic C++), oop_test.jsonl (500 mẫu OOP C++),
    unified_test.jsonl (1000 mẫu hợp nhất cả Basic + OOP), ood_c_test.jsonl và ood_cpp_test.jsonl
    (tập ngoại miền từ CodeMirage). Script đảm bảo phân tầng cân bằng (balanced stratified split)
    giữa nhãn AI và Human trong mỗi tập.

### create_test_sets.py
  - Loại: Python Script
  - Nguồn gốc: retrain ML models/
  - Mô tả: Phiên bản ban đầu (v1) của script tạo tập dữ liệu test. Đơn giản hơn bản improved, chỉ
    tạo các tập test cơ bản từ dữ liệu Basic C++ mà chưa tích hợp dữ liệu OOP và OOD. Được giữ lại
    để tham khảo lịch sử phát triển.

### generate_baselines_improved.py
  - Loại: Python Script (~13KB)
  - Nguồn gốc: retrain ML models/
  - Mô tả: Script xây dựng các mô hình nền tảng (baselines) để làm mốc so sánh hiệu năng. Script tính
    giá trị trung bình (mean) của từng đặc trưng cho nhóm mã AI và nhóm mã Human, lưu thành file
    baselines.json. Các baselines này được sử dụng trong giao diện web (app.py) để hiển thị thanh so
    sánh trực quan (radar chart) giữa giá trị đặc trưng của mã nguồn đang phân tích so với giá trị
    trung bình của AI và Human.

### inference_improved.py
  - Loại: Python Script (~17KB)
  - Nguồn gốc: retrain ML models/
  - Mô tả: Script suy luận (inference) chính thức trên 5 tập kiểm thử: basic_test, oop_test,
    unified_test, ood_c_test, ood_cpp_test. Đối với mỗi tập, script: (1) đọc file JSONL, (2) trích
    xuất 32 đặc trưng bằng CppFeatureExtractorV8, (3) chuẩn hóa bằng StandardScaler đã lưu, (4) chạy
    prediction bằng model LightGBM đã lưu, (5) tính Accuracy, F1-score, Precision, Recall, và in
    Confusion Matrix. Script cũng hỗ trợ so sánh hiệu năng giữa nhiều mô hình ML cùng lúc (LightGBM,
    XGBoost, Random Forest) để tạo bảng so sánh trong luận văn (Bảng 4.1).

### inference_ood.py
  - Loại: Python Script (~22KB)
  - Nguồn gốc: retrain ML models/
  - Mô tả: Script suy luận chuyên biệt dành riêng cho tập dữ liệu ngoại miền OOD (Out-of-Distribution).
    Script thực hiện đánh giá chi tiết hơn inference_improved.py bằng cách: (1) tách kết quả theo từng
    mô hình LLM sinh mã (GPT-4o-mini, DeepSeek-R1, Claude-3.5-Haiku, Gemini-2.0-Flash, Llama3.3-70B,
    Qwen2.5-Coder, O3-mini, v.v.) để xem mô hình nào dễ/khó phát hiện nhất, (2) đánh giá riêng biệt
    trên mã C và mã C++ để kiểm tra thiên lệch ngôn ngữ, (3) tính chỉ số TNR (True Negative Rate /
    Specificity) trên mẫu Human OOD — đây là chỉ số đánh giá khả năng thích nghi miền (Domain Adaptation)
    vì mã Human ngoại miền có phong cách rất khác so với dữ liệu huấn luyện.

### run_chapter4_experiments.py
  - Loại: Python Script
  - Nguồn gốc: tools/
  - Mô tả: MÃ NGUỒN CHẠY THỰC NGHIỆM CỦA LUẬN VĂN (BẢN CŨ). Script tự động hóa toàn bộ 5 thí
    nghiệm cốt lõi mô tả trong Chương 4 của luận văn tốt nghiệp, bao gồm:
    - Thực nghiệm 1: So sánh hiệu năng các mô hình ML (LightGBM, XGBoost, Random Forest) trên 3 tập
      kiểm thử ID (basic_test, oop_test, unified_test).
    - Thực nghiệm 2: Giải thích mô hình bằng SHAP (tính SHAP values, vẽ biểu đồ Bar và Beeswarm).
    - Thực nghiệm 3: Đánh giá GraphCodeBERT 1-Fold trên dữ liệu thực.
    - Thực nghiệm 4: Kết hợp mô hình lai (Hybrid) với alpha=0.48, so sánh ML vs DL vs Hybrid.
    - Thực nghiệm 5: Đánh giá OOD trên tập 12,599 files.
    LƯU Ý: Đây là bản thực nghiệm CŨ với alpha=0.48 và OOD 12,599 files. Bản thực nghiệm MỚI (bài
    báo khoa học) sử dụng run_ensemble_hybrid_experiment.py với alpha=0.50 và OOD 39,999 files.

### run_sanity_check_ensemble.py
  - Loại: Python Script
  - Nguồn gốc: tools/
  - Mô tả: Script kiểm tra tính tương thích (sanity check) của hệ thống mô hình Ensemble trước khi
    chạy thử nghiệm lớn. Script kiểm tra: (1) file predictions của 10-fold GraphCodeBERT có đầy đủ
    và đúng format không, (2) model LightGBM và scaler có load được không, (3) pipeline trích xuất
    đặc trưng có hoạt động đúng không, (4) kết quả dự đoán có nằm trong khoảng [0, 1] không. Mục đích:
    phát hiện sớm lỗi trước khi chạy thực nghiệm tốn thời gian trên toàn bộ 40,000 files OOD.

### visualize_pipeline.py
  - Loại: Python Script (~5KB)
  - Nguồn gốc: retrain ML models/
  - Mô tả: Script tự động vẽ các biểu đồ trực quan hóa phục vụ luận văn, bao gồm: (1) sơ đồ khối
    (block diagram) mô tả kiến trúc pipeline lọc đặc trưng 3 tầng (Correlation Filter -> SelectKBest
    -> Lasso L1), (2) biểu đồ cột (bar chart) thể hiện Feature Importance của top 20 đặc trưng quan
    trọng nhất từ mô hình LightGBM, (3) trang HTML tương tác hiển thị ma trận tương quan giữa các
    đặc trưng. Các biểu đồ đầu ra được lưu tại thư mục 4. Reports_and_Visualizations/.

================================================================================

## 3. MÔ HÌNH HỌC SÂU & MÔ HÌNH LAI (3. Deep_Learning_and_Hybrid/)

Lưu trữ các notebook và script thực nghiệm liên quan đến mô hình học sâu ngôn ngữ mã nguồn
(GraphCodeBERT — microsoft/graphcodebert-base) và thuật toán kết hợp lai (Hybrid Routing Model).


### Basic_of_LVTN_Graphcodebert_tính_micro_&_macro.ipynb
  - Loại: Jupyter Notebook (chạy trên Google Colab với GPU)
  - Nguồn gốc: Colab / Google Drive
  - Mô tả: Notebook CHÍNH huấn luyện mô hình học sâu GraphCodeBERT (microsoft/graphcodebert-base)
    trên bộ dữ liệu phát hiện mã AI. Quy trình huấn luyện:

    (1) Tiền xử lý dữ liệu: Tokenize mã nguồn C++ bằng RobertaTokenizer với max_length=512 tokens,
        truncation và padding tự động.
    (2) Kiểm thử chéo 10-fold Leave-One-Group-Out (LOGO): Chia dữ liệu thành 10 nhóm (groups) theo
        mô hình AI sinh mã (GPT-4o, Claude-3.5, DeepSeek, Gemini, v.v.). Mỗi fold giữ lại 1 nhóm
        làm tập test, 9 nhóm còn lại làm tập train. Điều này đảm bảo đánh giá cross-model
        generalization — tức mô hình phải phát hiện được mã từ một LLM mà nó chưa từng thấy.
    (3) Fine-tuning: Huấn luyện 3 epochs/fold, learning rate 2e-5, batch size 16, AdamW optimizer.
    (4) Tính toán chi tiết Macro F1 (trung bình F1 cả 2 class AI và Human) và Micro F1 (tổng hợp
        TP/FP/FN toàn bộ) cho từng fold.
    (5) Lưu predictions của mỗi fold ra file JSON để sử dụng cho Ensemble trong bước Hybrid.
    (6) Kết quả 10-fold: Macro F1 trung bình = 0.9290, cho thấy GraphCodeBERT có khả năng tổng quát
        tốt trên nhiều mô hình AI khác nhau.

### Hybrid_model.ipynb
  - Loại: Jupyter Notebook
  - Nguồn gốc: Colab / Google Drive
  - Mô tả: Notebook thực nghiệm lai ghép (Hybrid Routing Model) BẢN ĐẦU. Notebook thực hiện:
    (1) Load predictions từ GraphCodeBERT 1-Fold (không phải Ensemble) và LightGBM Standalone.
    (2) Thử nghiệm công thức kết hợp: P_hybrid = alpha * P_BERT + (1 - alpha) * P_LGBM.
    (3) Grid Search thủ công tìm alpha tối ưu, kết quả: alpha = 0.48.
    (4) So sánh hiệu năng Hybrid vs ML-only vs DL-only trên unified_test.
    LƯU Ý: Đây là bản thí nghiệm CŨ với alpha=0.48. Bản MỚI (bài báo) dùng 10-fold Ensemble DL
    và cho ra alpha=0.50 (xem run_ensemble_hybrid_experiment.py).

### run_ensemble_hybrid_experiment.py
  - Loại: Python Script (~600 dòng)
  - Nguồn gốc: tools/
  - Mô tả: MÃ NGUỒN CHẠY THỰC NGHIỆM CHÍNH CỦA BÀI BÁO KHOA HỌC (BẢN MỚI NHẤT). Đây là script
    toàn diện nhất, tích hợp tất cả các thí nghiệm cập nhật. Chi tiết:

    PHẦN A — Tổ hợp Ensemble DL:
      Đọc file dự đoán (predictions) từ 10 fold GraphCodeBERT (đã huấn luyện trong notebook ở trên).
      Tính xác suất trung bình Ensemble: S_DL = (1/10) * Σ P_fold_i. Phương pháp này giúp giảm
      phương sai dự đoán và tăng tính ổn định so với dùng 1 fold đơn lẻ.

    PHẦN B — Grid Search tìm Alpha tối ưu:
      Thực hiện quét alpha từ 0.00 đến 1.00 (bước 0.02) trên tập unified_test:
        P_hybrid(alpha) = alpha * S_DL + (1 - alpha) * S_ML
      Tại mỗi giá trị alpha, tính Accuracy và F1-score. Kết quả: alpha* = 0.50 cho hiệu năng
      cao nhất 95.90% Accuracy / 0.9590 F1-score. Ý nghĩa: nhánh DL và nhánh ML đóng góp ngang
      bằng nhau trong mô hình lai cuối cùng.

    PHẦN C — Đánh giá In-Distribution (ID):
      Chạy mô hình Hybrid với alpha=0.50 trên 3 tập kiểm thử ID:
        - basic_test (500 mẫu Basic C++): 93.80%
        - oop_test (500 mẫu OOP C++): 98.00%
        - unified_test (1000 mẫu hợp nhất): 95.90%
      So sánh với LightGBM standalone (87.90%) và GraphCodeBERT Ensemble (92.90%).

    PHẦN D — Đánh giá Out-of-Distribution (OOD) trên 39,999 files:
      Chạy trên 2 tập OOD hoàn toàn mới, không trùng phân phối với dữ liệu huấn luyện:
        - OOD C (20,000 files: 10,000 AI từ 10 LLMs + 10,000 Human)
        - OOD C++ (19,999 files: ~10,000 AI từ 10 LLMs + ~10,000 Human)
      Dữ liệu lấy từ CodeMirage, bao gồm mã từ: GPT-4o-mini, DeepSeek-R1, DeepSeek-V3,
      Claude-3.5-Haiku, Gemini-2.0-Flash, Gemini-2.0-Flash-Thinking, Gemini-2.0-Pro, Llama3.3-70B,
      O3-mini, Qwen2.5-Coder.

    PHẦN E — Đánh giá Thích nghi miền (Domain Adaptation):
      So sánh trực tiếp LightGBM Cũ (chưa thích nghi) vs LightGBM Mới (retrain với 2,000 mẫu OOD
      đại diện) trên tập OOD. Kết quả nổi bật: LightGBM Mới tăng vọt từ 22.53% lên 88.01% trên
      tập OOD C++ — chứng minh hiệu quả vượt trội của chiến lược thích nghi miền.

    PHẦN F — Phân tích Detection Rate theo từng LLM:
      Tính tỷ lệ phát hiện AI (AI Detection Rate) cho từng mô hình LLM trên cả tập OOD C và C++.
      Xuất bảng chi tiết để chèn trực tiếp vào bài báo (Bảng XVI, XVII, XVIII).

    PHẦN G — Trực quan hóa:
      Tự động vẽ 3 biểu đồ: accuracy_vs_alpha.png (đường Grid Search), dl_vs_ml_vs_hybrid.png
      (so sánh cột), ood_ai_detection_rates.png (detection rate theo LLM).

================================================================================

## 4. BÁO CÁO THỰC NGHIỆM & TRỰC QUAN HÓA (4. Reports_and_Visualizations/)

Nơi lưu trữ các tài liệu báo cáo thực nghiệm chi tiết, bảng số liệu và các biểu đồ trực quan từ
thực nghiệm phục vụ việc chèn trực tiếp vào luận văn tốt nghiệp và bài báo khoa học.


### ensemble_thesis_report.md
  - Loại: Báo cáo Markdown
  - Mô tả: BÁO CÁO THỰC NGHIỆM MỚI NHẤT (bài báo khoa học). Tổng hợp toàn bộ kết quả thực nghiệm
    cập nhật từ run_ensemble_hybrid_experiment.py. Nội dung bao gồm:
    - Bảng 4.4: Hiệu năng Hybrid ID với alpha=0.50 trên basic_test/oop_test/unified_test.
    - Bảng 4.5: Hiệu năng OOD tổng thể trên 39,999 files (OOD C và OOD C++).
    - Bảng 4.5b: Chỉ số TNR (Specificity) trên 500 mẫu Human OOD — đánh giá Domain Adaptation.
    - Bảng 4.6 & 4.7: Detection Rate chi tiết cho từng mô hình LLM trên OOD C và C++.

### Feature_Extractor_Report.md
  - Loại: Báo cáo Markdown
  - Mô tả: Tài liệu kỹ thuật chi tiết mô tả đầy đủ 32 đặc trưng tĩnh trong bộ CppFeatureExtractorV8.
    Cho mỗi đặc trưng, báo cáo cung cấp: (1) tên kỹ thuật, (2) nhóm phân loại (Layout & Formatting,
    Naming Conventions, Structural Complexity, Habits & Idioms, Information Theory, OOP Metrics),
    (3) công thức toán học tính toán, (4) ý nghĩa phân biệt giữa mã AI và mã Human, (5) miền giá trị
    và đơn vị. Báo cáo cũng giải thích các chỉ số Halstead (Volume, Difficulty, Effort, Bugs) và
    chỉ số Maintainability Index (MI).

### optimal_k_report.md
  - Loại: Báo cáo Markdown
  - Mô tả: Báo cáo kết quả chi tiết từ việc chạy find_optimal_k.py. Trình bày bảng số liệu hiệu năng
    (Accuracy, F1) của 4 mô hình ML tại mỗi giá trị k (từ 5 đến 40). Giải thích lý do chọn k=33
    làm điểm tối ưu: tại k=33, mô hình LightGBM đạt F1 cao nhất; sau khi Lasso L1 loại 1 đặc trưng,
    còn lại 32 đặc trưng thực tế. Tăng k vượt 33 không cải thiện đáng kể hiệu năng nhưng tăng nguy
    cơ overfitting.

### accuracy_vs_alpha.png
  - Loại: Biểu đồ PNG
  - Mô tả: Đồ thị đường (line plot) kết quả Grid Search quét hệ số lai alpha từ 0.00 đến 1.00. Trục
    x = alpha, trục y = Accuracy (%) trên unified_test. Đồ thị cho thấy đỉnh hiệu năng tại alpha=0.50.
    Tương ứng với Hình 12 trong luận văn.

### dl_vs_ml_vs_hybrid.png
  - Loại: Biểu đồ PNG
  - Mô tả: Đồ thị cột nhóm (grouped bar chart) so sánh trực quan hiệu năng giữa 3 phương pháp:
    LightGBM (ML only), GraphCodeBERT Ensemble (DL only), và Hybrid Model (đề xuất) trên 3 tập kiểm
    thử ID. Cho thấy Hybrid luôn vượt trội hoặc ngang bằng so với từng nhánh riêng lẻ.

### ml_model_comparison.png
  - Loại: Biểu đồ PNG
  - Mô tả: Đồ thị cột nhóm so sánh trực quan Accuracy (%) giữa các mô hình học máy: LightGBM Cũ
    (44 đặc trưng), LightGBM Mới (32 đặc trưng + retrained), XGBoost, Random Forest trên các tập
    basic_test, oop_test, unified_test. Tương ứng với Hình 13 trong luận văn (Bảng 4.1 — Thực nghiệm 1).

### shap_summary_bar.png
  - Loại: Biểu đồ PNG
  - Mô tả: Biểu đồ cột ngang SHAP Summary (Bar plot). Thể hiện mean absolute SHAP value (trung bình
    trị tuyệt đối giá trị SHAP) của top 20 đặc trưng ảnh hưởng lớn nhất đến quyết định dự đoán
    AI/Human của LightGBM. Đặc trưng có SHAP value cao nhất = quan trọng nhất cho mô hình.
    Tương ứng với Hình 14 trong luận văn (Thực nghiệm 2 — XAI bằng SHAP).

### shap_summary_beeswarm.png
  - Loại: Biểu đồ PNG
  - Mô tả: Biểu đồ Beeswarm SHAP. Mỗi chấm đại diện cho 1 mẫu dữ liệu; vị trí ngang = giá trị
    SHAP (dương = đẩy về phía AI, âm = đẩy về phía Human); màu sắc = giá trị thực của đặc trưng
    (đỏ = cao, xanh = thấp). Cho phép đọc được chiều hướng tác động cụ thể (VD: whitespace_entropy
    thấp → mã AI; avg_identifier_length cao → mã AI; brace_style_consistency cao → mã AI).
    Tương ứng với Hình 15 trong luận văn (Thực nghiệm 2 — XAI bằng SHAP).

### ood_ai_detection_rates.png
  - Loại: Biểu đồ PNG
  - Mô tả: Biểu đồ cột/nhóm trực quan hóa tỷ lệ phát hiện mã AI (AI Detection Rate) trên tập OOD
    cho từng mô hình LLM (GPT-4o-mini, DeepSeek-R1, Claude-3.5-Haiku, v.v.). Tách riêng kết quả
    cho OOD C và OOD C++. Tương ứng với Bảng 4.6 & 4.7 trong luận văn (Thực nghiệm 5).

### optimal_k_search.png
  - Loại: Biểu đồ PNG
  - Mô tả: Đồ thị đường thể hiện F1-score của 4 mô hình ML (LightGBM, XGBoost, Random Forest,
    Logistic Regression) tương ứng với số lượng đặc trưng đầu vào k (từ 5 đến 40). Đồ thị cho thấy
    hiệu năng tăng nhanh khi k tăng từ 5-15, sau đó bão hòa/giảm nhẹ sau k=33, chứng minh việc
    lựa chọn k=33 là điểm cân bằng tối ưu giữa hiệu năng và chi phí tính toán.

### feature_selection_pipeline.png
  - Loại: Sơ đồ khối PNG
  - Mô tả: Sơ đồ khối (block diagram) mô tả trực quan kiến trúc pipeline lọc đặc trưng 3 tầng:
    [44 đặc trưng gốc] → Tầng 1 (Correlation r>0.85) → Tầng 2 (SelectKBest MI, k=33) → Tầng 3
    (Lasso L1) → [32 đặc trưng tối ưu]. Dùng chèn vào Chương 3 (Phương pháp) của luận văn.

### feature_selection_visualization.html
  - Loại: Trang HTML tương tác (Plotly)
  - Mô tả: Trang web tương tác hiển thị: (1) ma trận tương quan (correlation matrix heatmap) giữa
    các đặc trưng trước và sau lọc, (2) biểu đồ scatter hiển thị Mutual Information score của từng
    đặc trưng, (3) danh sách đặc trưng bị loại tại mỗi tầng. Cho phép hover, zoom, lọc tương tác
    để khám phá dữ liệu. Mở file bằng trình duyệt web.

================================================================================

## 5. DỮ LIỆU (5. Datasets/) — CẦN XÁC NHẬN TRƯỚC KHI COPY

Danh sách các tập dữ liệu đã sử dụng trong thực nghiệm (xem mục báo cáo Dataset bên dưới README
này để xác nhận trước khi copy vào thư mục tong hop).
