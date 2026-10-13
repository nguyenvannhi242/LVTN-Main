# 🎓 Phát Hiện Mã Nguồn C++ Do AI Sinh Ra Bằng Mô Hình Học Sâu & Học Máy Giải Thích Được (XAI)

> **Khóa luận tốt nghiệp & Nghiên cứu Khoa học**  
> *Đề tài: Nghiên cứu giải pháp phát hiện lạm dụng trí tuệ nhân tạo trong bài tập lập trình C++ bằng mô hình học sâu và học máy giải thích được*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving%20SSE-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Django](https://img.shields.io/badge/Django-5.x%20Dashboard-092E20.svg?logo=django)](https://www.djangoproject.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-32%20Features-green.svg)](https://lightgbm.readthedocs.io/)
[![GraphCodeBERT](https://img.shields.io/badge/GraphCodeBERT-10--Fold%20Ensemble-orange.svg)](https://huggingface.co/microsoft/graphcodebert-base)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20Gold%20Layer-4169E1.svg?logo=postgresql)](https://www.postgresql.org/)

---

## 📌 Mục Lục
1. [Giới Thiệu Đề Tài & Mục Tiêu](#-giới-thiệu-đề-tài--mục-tiêu)
2. [Đóng Góp Phương Pháp Luận Cốt Lõi](#-đóng-góp-phương-pháp-luận-cốt-lõi)
3. [Kết Quả Thực Nghiệm Nổi Bật](#-kết-quả-thực-nghiệm-nổi-bật)
4. [Kiến Trúc Nền Tảng & Hệ Thống MLOps](#-kiến-trúc-nền-tảng--hệ-thống-mlops)
5. [Cấu Trúc Thư Mục Repository](#-cấu-trúc-thư-mục-repository)
6. [Giao Diện & Tính Năng XAI (Explainable AI)](#-giao-diện--tính-năng-xai-explainable-ai)
7. [Hướng Dẫn Cài Đặt & Khởi Chạy Nhanh](#-hướng-dẫn-cài-đặt--khởi-chạy-nhanh)
8. [Tái Hiện Thực Nghiệm Nghiên Cứu](#-tái-hiện-thực-nghiệm-nghiên-cứu)
9. [Tài Liệu Kỹ Thuật Bổ Sung](#-tài-liệu-kỹ-thuật-bổ-sung)

---

## 📖 Giới Thiệu Đề Tài & Mục Tiêu

Sự bùng nổ của các mô hình ngôn ngữ lớn (LLMs như GPT-4o, Claude 3.5, DeepSeek, Gemini) khiến việc sinh mã nguồn giải bài tập lập trình trở nên phổ biến, đặt ra thách thức nghiêm trọng đối với giáo dục đại học trong việc đánh giá thực lực sinh viên.

Hệ thống này được nghiên cứu và xây dựng nhằm giải quyết hai bài toán lớn:
1. **Độ chính xác cao & Khả năng tổng quát hóa (Generalization)**: Phát hiện mã do AI sinh ra trên cả bài tập lập trình C++ cơ bản (Basic) lẫn lập trình hướng đối tượng (OOP), đồng thời chống chịu tốt trước mã ngoại miền (Out-of-Distribution - OOD) từ các LLM mới nhất.
2. **Khả năng giải thích minh bạch (Explainable AI - XAI)**: Cung cấp bằng chứng định lượng rõ ràng cho giảng viên (thẻ điểm phong cách lập trình, đối sánh đặc trưng SHAP, cảnh báo bất thường) nhằm tránh tối đa việc kết luận oan cho sinh viên.

---

## 💡 Đóng Góp Phương Pháp Luận Cốt Lõi

```text
               ┌────────────────────────────────────────────────────────┐
               │                  MÃ NGUỒN C++ ĐẦU VÀO                  │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                  ┌───────────────────────┴────────────────────────┐
                  ▼                                                ▼
     ┌─────────────────────────┐                      ┌─────────────────────────┐
     │   NHÁNH HỌC MÁY (ML)    │                      │   NHÁNH HỌC SÂU (DL)    │
     │  Bộ trích xuất 32 ĐT    │                      │     GraphCodeBERT       │
     │    (Stylometry & OOP)   │                      │  10-Fold LOGO Ensemble  │
     │      LightGBM Model     │                      │   Token Semantic & DFG  │
     └────────────┬────────────┘                      └────────────┬────────────┘
                  │ P_ML                                           │ P_DL
                  └───────────────────────┬────────────────────────┘
                                          ▼
                       ┌─────────────────────────────────────┐
                       │      MÔ HÌNH KẾT HỢP LAI (HYBRID)   │
                       │    P_final = 0.50*P_DL + 0.50*P_ML  │
                       │      Accuracy ID: 95.90% (F1: 0.959)│
                       └──────────────────┬──────────────────┘
                                          ▼
                       ┌─────────────────────────────────────┐
                       │     GIẢI THÍCH MINH BẠCH (XAI)      │
                       │ • SHAP Feature Attribution          │
                       │ • Contrastive Analysis (AI vs Human)│
                       │ • Behavioral Coding Scorecard       │
                       └─────────────────────────────────────┘
```

1. **Quy trình Lọc Đặc Trưng 3 Tầng (3-Stage Feature Selection Pipeline)**:
   - Từ 44 đặc trưng ban đầu (bố cục, định danh, độ phức tạp cấu trúc, thói quen viết mã C++, hướng đối tượng OOP), áp dụng:
     - **Tầng 1 (Pearson Correlation Filter)**: Loại bỏ đa cộng tuyến ($|r| > 0.85$).
     - **Tầng 2 (SelectKBest Mutual Information)**: Quét điểm uốn $k=5 \dots 40$, chọn $k=33$ tối ưu thông tin phi tuyến.
     - **Tầng 3 (Lasso L1 Regularization)**: Ép trọng số dư thừa về 0, giữ lại tập **32 đặc trưng tinh gọn nhất**.
2. **Mô Hình Học Sâu GraphCodeBERT với 10-Fold Leave-One-Group-Out (LOGO)**:
   - Phân tách theo từng mô hình AI sinh mã để huấn luyện và kiểm định chéo, đảm bảo mô hình không bị phụ thuộc vào phân phối của một LLM cụ thể.
3. **Thuật Toán Kết Hợp Lai Tối Ưu (Optimal Hybrid Routing)**:
   - Quét lưới (Grid Search) hệ số lai $\alpha \in [0.0, 1.0]$. Giá trị tối ưu được xác lập tại **$\alpha = 0.50$**, chứng minh sự đóng góp cân bằng hoàn hảo giữa nhánh ngữ nghĩa mã (GraphCodeBERT) và nhánh phong cách cú pháp (LightGBM).
4. **Chiến Lược Thích Nghi Miền (Domain Adaptation)**:
   - Bổ sung các mẫu đại diện ngoại miền giúp tỷ lệ phát hiện trên tập OOD tăng vượt bậc từ **22.53% lên 88.01%** trên tập C++.

---

## 📊 Kết Quả Thực Nghiệm Nổi Bật

### 1. Hiệu Năng Trên Các Tập Dữ Liệu Nội Miền (In-Distribution - ID)

| Phương pháp | Basic C++ (500 mẫu) | OOP C++ (500 mẫu) | Hợp nhất (1,000 mẫu) | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM Standalone** (32 đặc trưng) | 88.40% | 87.40% | 87.90% | 0.8789 |
| **GraphCodeBERT Ensemble** (10 Folds) | 89.20% | 96.60% | 92.90% | 0.9288 |
| 🏆 **Mô hình Lai đề xuất (Hybrid $\alpha=0.50$)** | **93.80%** | **98.00%** | **95.90%** | **0.9590** |

### 2. Kiểm Thử Ngoại Miền (Out-of-Distribution - OOD) Trên Gần 40,000 File

Được đánh giá trên tập dữ liệu ngoại miền khổng lồ từ **10 mô hình LLM hiện đại** (CodeMirage benchmark):

| Tập kiểm thử | Số lượng mẫu | Tỷ lệ phát hiện AI (Sau Domain Adaptation) | True Negative Rate (Human Specificity) |
| :--- | :---: | :---: | :---: |
| **OOD C++** | 19,999 files | **88.01%** | **98.00%** (TNR trên mẫu Human) |
| **OOD C** | 20,000 files | **83.45%** | **95.60%** (TNR trên mẫu Human) |

---

## 🏛️ Kiến Trúc Nền Tảng & Hệ Thống MLOps

Hệ thống được thiết kế theo tiêu chuẩn công nghiệp nhằm phục vụ triển khai thực tế trong nhà trường:

```text
┌────────────────────────────────────────────────────────────────────┐
│              🖥️  LOCAL MACHINE (Laptop / PC)                       │
│                                                                    │
│  ┌────────────────────────┐      ┌────────────┐  ┌────────────────┐ │
│  │  🌐 DJANGO WEB APP     │      │ Redis/     │  │ Celery Workers │ │
│  │  (Auth, Dash, Bronze)  │      │ Redpanda   │  │ (ETL, S3 Push) │ │
│  └───────────┬────────────┘      └─────┬──────┘  └───────┬────────┘ │
│              │                         │                 │         │
│              └───────────────┬─────────┴─────────────────┘         │
│                              │                                     │
│  ┌────────────────────────┐  │   PostgreSQL (Gold) ◄─────┘         │
│  │ ⚡ FastAPI Orchestrator│◄─┘                                     │
│  │    (Port 8001)         │           DagsHub S3 (Data Lake) ──┐   │
│  └───────────┬────────────┘                                    │   │
└──────────────┼─────────────────────────────────────────────────│───┘
               │ HTTPS (ngrok tunnel) hoặc localhost             │
               ▼                                                 │
┌────────────────────────────────────────────────────────────────│───┐
│              ☁️ GPU RUNTIME (Local GPU hoặc Google Colab)       │   │
│                                                                │   │
│  ┌──────────────────────────────────────────────────────────┐  │   │
│  │  🔧 LOCAL GPU SERVER (Port 8002) HOẶC COLAB FASTAPI       │  │   │
│  │  LangGraph Agent │ GraphCodeBERT Ensemble │ Qwen 2.5 Coder│  │   │
│  └──────────────────────────────────────────────────────────┘  │   │
└────────────────────────────────────────────────────────────────│───┘
```

- **Kiến trúc Dữ liệu Medallion (Data Lakehouse)**:
  - 🥉 **Bronze Layer (DagsHub S3 + PostgreSQL)**: Lưu trữ mã nguồn thô của sinh viên dưới dạng JSONL bất biến.
  - 🥈 **Silver Layer (DagsHub S3)**: Bảng đặc trưng đã tiền xử lý (32 đặc trưng số + 512 token IDs) dưới định dạng `.parquet`.
  - 🥇 **Gold Layer (PostgreSQL)**: Kết quả suy luận, giải thích XAI, dữ liệu phân tích phục vụ Dashboard.
- **Microservices & Công nghệ sử dụng**:
  - **Django 5.x**: Web portal cho giảng viên, quản lý người dùng, xem báo cáo chi tiết.
  - **FastAPI**: Serving AI microservice hỗ trợ Server-Sent Events (SSE) để truyền kết quả streaming thời gian thực.
  - **Celery & Redis**: Xử lý background jobs (ETL, cập nhật chỉ số ngầm).
  - **LangGraph Multi-Agent**: Điều phối 4 nút xử lý (*Router → Analyzer → Judge → Critique*).

---

## 📁 Cấu Trúc Thư Mục Repository

Kho lưu trữ được cấu trúc rõ ràng, tách bạch giữa mã nguồn hệ thống sản phẩm và mã nguồn nghiên cứu khoa học:

```text
LVTN-Main/
├── README.md                                 # Tài liệu tổng quan dự án (tài liệu này)
├── ARCHITECTURE_MAP.md                       # Bản đồ định tuyến kiến trúc chi tiết
├── SETUP_AND_RUN.md                          # Cẩm nang cài đặt và chạy hệ thống từng bước
├── requirements.txt                          # Toàn bộ thư viện phụ thuộc Python
├── docker-compose.yml                        # Docker stack: PostgreSQL, Redis, Redpanda
├── Makefile / dev.sh                         # Lệnh tự động hóa khởi động môi trường dev
├── manage.py / app.py                        # Entrypoints chạy Django Web và FastAPI
│
├── Tong hop script/                          # 🧪 MÃ NGUỒN NGHIÊN CỨU & THỰC NGHIỆM TỔNG HỢP
│   ├── README.md                             # Hướng dẫn chi tiết 5 phần thực nghiệm
│   ├── 1. Feature_Selection_and_Extraction/  # Lọc đặc trưng 3 tầng (find_optimal_k.py)
│   ├── 2. ML_Training_and_Inference/         # Huấn luyện ML, CV 5-fold, SHAP, inference
│   ├── 3. Deep_Learning_and_Hybrid/          # GraphCodeBERT 10-fold, Grid Search Alpha
│   ├── 4. Reports_and_Visualizations/        # Báo cáo Markdown & biểu đồ nghiên cứu
│   └── 5. Datasets/                          # Các tập dữ liệu benchmark JSONL chuẩn hóa
│
├── data_splits/                              # 📊 TẬP DỮ LIỆU CHUẨN HÓA (JSONL)
│   ├── train_set/                            # Tập train Basic C++ & OOP C++
│   ├── test_set_ID/                          # Tập test nội miền (basic_test, oop_test)
│   └── test_set_unified/                     # Tập test thống nhất 1,000 mẫu
│
├── src/                                      # 🚀 MÃ NGUỒN HỆ THỐNG CHÍNH (ENTERPRISE BACKEND)
│   ├── django_web/                           # Web App Django & MLOps Admin Dashboard
│   ├── fastapi_service/                      # FastAPI Orchestrator, AI Engine & XAI
│   ├── celery_workers/                       # Background Jobs & Medallion ETL Pipeline
│   ├── data_pipeline/                        # Pipeline Bronze -> Silver -> Gold
│   ├── local_gpu_server/                     # Standalone GPU Inference Server
│   ├── local_gpu_agent/                      # Local Hybrid Agent Runtime
│   └── shared/                               # Config, Database sessions, Contracts, S3
│
├── System architecture/                      # 📐 TÀI LIỆU THIẾT KẾ KIẾN TRÚC KỸ THUẬT
│   ├── 00_system_overview.md                 # Tổng quan toàn hệ thống
│   ├── 01_mlops_pipeline.md                  # Chi tiết pipeline MLOps & LangGraph
│   ├── 02_admin_dashboard.md                 # Thiết kế các màn hình quản trị
│   ├── 03_shared_infrastructure.md           # Thiết kế hạ tầng dùng chung
│   └── 04_operations.md                      # Vận hành, giám sát và xử lý sự cố
│
├── infrastructure/                           # Kubernetes manifests & Ansible playbooks
├── monitoring/                               # Cấu hình Prometheus & Alerting Rules
├── design_mockups/                           # Bản mẫu giao diện HTML tĩnh
└── tests/                                    # Unit tests, integration tests & E2E tests
```

---

## 🎨 Giao Diện & Tính Năng XAI (Explainable AI)

Giao diện phân tích hỗ trợ đắc lực cho người chấm bài thông qua 3 bảng điều khiển trực quan:

1. **Thẻ Điểm Hành Vi Lập Trình (Behavioral Coding Scorecard)**:
   - Nhóm 32 đặc trưng thành 5 khía cạnh lập trình:
     - 📐 *Bố cục & Thụt lề (Layout & Formatting)*
     - 🏷️ *Quy ước Đặt tên (Naming Conventions)*
     - 🧩 *Độ phức tạp Cấu trúc (Structural Complexity)*
     - ☕ *Thói quen Viết mã C++ (Coding Habits)*
     - 🏛️ *Cấu trúc Hướng đối tượng (OOP Structure)*
   - Đánh giá độ thiên lệch theo thang điểm phần trăm trực quan.
2. **Bảng Đối Sánh Tương Phản (Contrastive XAI Panel)**:
   - Hiển thị giá trị thực tế của mã nguồn sinh viên đặt cạnh khoảng giá trị đặc trưng thông thường của Người và AI để chỉ ra điểm dị biệt.
3. **Huy Hiệu Cảnh Báo Xung Đột (Conflict Badges)**:
   - Tự động phát hiện và cảnh báo khi có sự mâu thuẫn giữa phong cách cục bộ (vd: đặt tên chuẩn AI nhưng cấu trúc lồng nhau kiểu sinh viên).

---

## ⚡ Hướng Dẫn Cài Đặt & Khởi Chạy Nhanh

### 1. Yêu cầu môi trường
- Hệ điều hành: Windows / Linux / macOS
- **Python**: 3.10 hoặc 3.11
- **Docker & Docker Compose** (để chạy cơ sở dữ liệu và message broker)

### 2. Cài đặt các thư viện cần thiết
```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
source .venv/bin/activate    # Trên Linux/macOS
# .venv\Scripts\activate     # Trên Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 3. Khởi chạy hệ thống cơ sở hạ tầng (Docker)
```bash
docker-compose up -d
```
Lệnh này sẽ khởi chạy:
- **PostgreSQL 16** (Port 5432)
- **Redis** (Port 6379)
- **Redpanda / Kafka** (Port 9092)

### 4. Khởi chạy Ứng dụng Web & API Serving
```bash
# Cách 1: Chạy toàn bộ bằng script tự động (Linux/macOS với tmux)
make dev
# Hoặc: ./dev.sh

# Cách 2: Chạy độc lập từng dịch vụ (Windows / Manual)
# Terminal 1: Chạy Django Web Dashboard
python manage.py migrate
python manage.py runserver 0.0.0.0:8000

# Terminal 2: Chạy FastAPI AI Serving
uvicorn src.fastapi_service.main:app --host 0.0.0.0 --port 8001 --reload
```

- **Giao diện Người dùng & Phân tích**: [http://localhost:8000](http://localhost:8000)
- **Bảng điều khiển Quản trị MLOps**: [http://localhost:8000/dashboard/](http://localhost:8000/dashboard/)
- **API Swagger Documentation**: [http://localhost:8001/docs](http://localhost:8001/docs)

---

## 🔬 Tái Hiện Thực Nghiệm Nghiên Cứu

Tất cả các thực nghiệm trong đề tài đều được đóng gói độc lập trong thư mục [`Tong hop script/`](file:///c:/Users/2/Desktop/WORKSPACE/LVTN-Main/Tong%20hop%20script):

### 1. Tuyển chọn đặc trưng tối ưu (Feature Selection)
```bash
python "Tong hop script/1. Feature_Selection_and_Extraction/find_optimal_k.py"
```
*Tự động thực hiện quy trình 3 tầng (Correlation -> Mutual Information -> Lasso L1), xuất kết quả ra file markdown và vẽ biểu đồ.*

### 2. Huấn luyện mô hình ML & Đánh giá Stratified K-Fold
```bash
python "Tong hop script/2. ML_Training_and_Inference/retrain_improved.py"
```
*Chạy kiểm định chéo 5-fold trên LightGBM, XGBoost, Random Forest, tính toán SHAP values và lưu trữ scaler cùng danh sách 32 đặc trưng.*

### 3. Chạy thực nghiệm Mô hình Lai (Ensemble DL + ML Hybrid)
```bash
python "Tong hop script/3. Deep_Learning_and_Hybrid/run_ensemble_hybrid_experiment.py"
```
*Thực hiện Grid Search tìm hệ số alpha, đánh giá so sánh ID/OOD và xuất toàn bộ các bảng số liệu tương ứng.*

---

## 📚 Tài Liệu Kỹ Thuật Bổ Sung

Để tìm hiểu sâu hơn về kiến trúc kỹ thuật nội bộ, vui lòng tham khảo các tài liệu liên kết:
- 🗺️ [**ARCHITECTURE_MAP.md**](file:///c:/Users/2/Desktop/WORKSPACE/LVTN-Main/ARCHITECTURE_MAP.md): Bản đồ định vị và điều hướng toàn bộ module.
- ⚙️ [**SETUP_AND_RUN.md**](file:///c:/Users/2/Desktop/WORKSPACE/LVTN-Main/SETUP_AND_RUN.md): Hướng dẫn chi tiết cấu hình biến môi trường và chạy cụm dịch vụ.
- 📐 [**System architecture/**](file:///c:/Users/2/Desktop/WORKSPACE/LVTN-Main/System%20architecture): Bộ tài liệu 5 phần về kiến trúc tổng thể, MLOps, Dashboard, Hạ tầng và Vận hành.
- 🧪 [**Tong hop script/README.md**](file:///c:/Users/2/Desktop/WORKSPACE/LVTN-Main/Tong%20hop%20script/README.md): Báo cáo chi tiết từng script nghiên cứu và giải thích công thức các đặc trưng.

---

<div align="center">
  <sub>Đồ án Tốt nghiệp & Công trình Nghiên cứu Khoa học — Phát triển với tính minh bạch, chuẩn mực và khả năng ứng dụng thực tiễn cao.</sub>
</div>
