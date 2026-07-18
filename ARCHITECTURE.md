# Kiến trúc Hệ thống: Fraud-XAI Anomaly Detection Framework

Tài liệu này trình bày tổng quan về kiến trúc và luồng xử lý (pipeline) từ đầu đến cuối của hệ thống Fraud-XAI trong việc phát hiện gian lận thẻ tín dụng dưới tác động của mất cân bằng dữ liệu.

## Tổng quan Kiến trúc

```mermaid
flowchart TD
    %% Define styles
    classDef ingest fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px;
    classDef process fill:#fff3e0,stroke:#fb8c00,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#43a047,stroke-width:2px;
    classDef eval fill:#fce4ec,stroke:#d81b60,stroke-width:2px;
    classDef dashboard fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;

    %% Data Ingestion & Preprocessing
    subgraph S1 [1. Thu thập & Tiền xử lý Dữ liệu]
        A[Raw Data: creditcard.csv] --> B(Tách Train/Test)
        B --> C[Chuẩn hóa - Standard Scaler]
        C --> D[(Dữ liệu Train đã xử lý)]
        C --> E[(Dữ liệu Test đã xử lý)]
    end
    class S1,A,B,C,D,E ingest;

    %% Imbalance Handling
    subgraph S2 [2. Xử lý Mất cân bằng Dữ liệu]
        D --> F1[C0: Class-weighting]
        D --> F2[C1: SMOTE]
        D --> F3[C2: SMOTE-ENN]
    end
    class S2,F1,F2,F3 process;

    %% Model Training
    subgraph S3 [3. Huấn luyện Mô hình]
        F1 --> M1[Random Forest]
        F2 --> M2[XGBoost]
        F3 --> M3[CatBoost]
        F1 -.-> M4[Logistic Regression]
        F2 -.-> M4
        F3 -.-> M4
    end
    class S3,M1,M2,M3,M4 model;

    %% Evaluation
    subgraph S4 [4. Đánh giá]
        M1 --> P[Chỉ số Hiệu suất \n PR-AUC, F1, Recall, ROC-AUC]
        M2 --> P
        M3 --> P
        M4 --> P
        
        P --> JSON[(results/*.json)]
    end
    class S4,P,JSON eval;

    %% Dashboarding
    subgraph S5 [5. Streamlit Dashboard]
        JSON --> UI1[Panel Hiệu suất]
        JSON --> UI2[Biểu đồ Heatmap]
        JSON --> UI3[Phân tích Đánh đổi Scatter]
    end
    class S5,UI1,UI2,UI3 dashboard;

    %% Connections across subgraphs
    E --> P
```
```

## Các Bước Chi Tiết Trong Pipeline

### 1. Thu thập & Tiền xử lý Dữ liệu (`src/preprocess.py`)
- **Tập dữ liệu:** Sử dụng bộ dữ liệu Kaggle `creditcard.csv` (mất cân bằng cực cao: chỉ có 0.17% là gian lận).
- **Phân chia:** Hàm `load_and_split()` tách dữ liệu theo tỷ lệ 80/20 có phân tầng (stratification) để giữ nguyên tỷ lệ phân bổ nhãn.
- **Chuẩn hóa (Scaling):** Hàm `scale_features()` áp dụng `StandardScaler` lên các đặc trưng `Amount` và `Time` (và các đặc trưng `V1-V28` khác nếu cần), đảm bảo phân phối có trung bình bằng 0 và phương sai bằng 1. Tập kiểm thử (test set) được chuẩn hóa nghiêm ngặt dựa trên các thông số thống kê từ tập huấn luyện (train set).

### 2. Xử lý Mất cân bằng Dữ liệu (`src/imbalance.py`)
Để giải quyết vấn đề mất cân bằng lớp dữ liệu nghiêm trọng, ba điều kiện thử nghiệm riêng biệt được áp dụng cho tập dữ liệu huấn luyện:
- **C0 (Class-weighting):** Dữ liệu huấn luyện được giữ nguyên trạng thái mất cân bằng gốc. Sự mất cân bằng này được xử lý trực tiếp bên trong các thuật toán máy học thông qua việc tinh chỉnh siêu tham số (ví dụ: `scale_pos_weight` trong XGBoost, `class_weight='balanced'` trong RF và LR).
- **C1 (SMOTE):** Kỹ thuật Oversampling lấy mẫu thiểu số tổng hợp (Synthetic Minority Over-sampling Technique) được sử dụng để tổng hợp ra các mẫu gian lận mới, bắt buộc tỷ lệ đạt 1:1.
- **C2 (SMOTE-ENN):** Kỹ thuật lai ghép sử dụng SMOTE để oversample và sau đó áp dụng Edited Nearest Neighbors (ENN) để dọn dẹp các mẫu nhiễu/nằm ở vùng biên, giúp tạo ra các ranh giới quyết định (decision boundaries) rõ ràng hơn.
*(Lưu ý: Tập kiểm thử (Test set) không bao giờ bị can thiệp resample, nhằm bảo tồn phân phối dữ liệu thực tế).*

### 3. Huấn luyện Mô hình (`src/models.py`)
Bốn mô hình tạo thành nòng cốt của các thử nghiệm:
- **Random Forest (RF):** Đóng vai trò là mô hình ensemble (tổ hợp) cơ bản, cực kỳ bền vững và ổn định.
- **XGBoost (XGB):** Mô hình Tree-based boosting, nổi tiếng với hiệu suất cao nhưng thường nhạy cảm với nhiễu.
- **CatBoost:** Mô hình boosting tiên tiến giúp cân bằng giữa độ chính xác dự đoán và tính minh bạch/giải thích.
- **Logistic Regression (LR):** Mô hình tuyến tính cơ sở (baseline) dùng để so sánh chéo giữa các họ thuật toán.

Mỗi mô hình được đánh giá dưới cả 3 điều kiện xử lý mất cân bằng (tổng cộng 12 cấu hình). Các siêu tham số (hyperparameters) được khóa chặt ở các giá trị tối ưu đã được tinh chỉnh để đảm bảo sự công bằng khi so sánh hiệu năng.

### 4. Đánh giá (`src/evaluate.py`)
- **Đánh giá Hiệu suất:** Đối với mỗi mẫu thử nghiệm, điểm xác suất sẽ được tính toán và chuyển đổi thành các chỉ số hiệu suất: `PR-AUC`, `ROC-AUC`, `Recall`, và `F1`. Trong đó `PR-AUC` đóng vai trò là thước đo hiệu suất chính.

### 5. Streamlit Dashboard Tương tác (`dashboard/app.py`)
- **Tab 1 (Performance):** Biểu đồ Bar chart động để so sánh các chỉ số PR-AUC, F1-Score, Recall và ROC-AUC giữa các cấu hình.
- **Tab 2 (Detailed Analysis):** Biểu đồ Heatmap hiển thị sự chênh lệch hiệu năng và biểu đồ Scatter phân tích sự đánh đổi giữa PR-AUC và F1.
- **Tab 3 (Full Results Table):** Bảng tổng hợp đầy đủ kết quả chỉ số kèm theo tính năng tải về (Download) file CSV.
