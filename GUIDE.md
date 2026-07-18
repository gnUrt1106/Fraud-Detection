Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường, chạy hyperparameter tuning (Optuna), chạy thí nghiệm chính, và trực quan hóa kết quả trên Streamlit Dashboard.

---

## 1. Thiết lập Môi trường (Local)
Dự án yêu cầu **Python 3.12** để tránh các lỗi không tương thích phiên bản.

```bash
# Tạo môi trường ảo
python3.12 -m venv .venv

# Kích hoạt môi trường ảo
source .venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 2. Tune Hyperparameters bằng Optuna (Kaggle)
Để tránh quá tải máy cá nhân, hãy chạy tuning trên Kaggle. Clone code mới nhất và chia thành 2 phiên chạy (Notebooks) khác nhau để tối ưu quota GPU:

* **Phiên GPU (Chọn GPU P100 hoặc T4x2 làm Accelerator):**
  Dùng để tune các mô hình boosting hạng nặng: `XGBClassifier` và `CatBoostClassifier`.
  ```bash
  !git clone https://github.com/gnUrt1106/Fraud-XAI.git
  %cd Fraud-XAI
  !pip install -r requirements.txt
  !python tune_hyperparams.py --model gpu --trials 50 --patience 10
  ```
  *(Các mô hình sẽ được tăng tốc bằng GPU CUDA. Lệnh sẽ tự động dừng nếu sau 10 trials liên tiếp không cải thiện PR-AUC).*

* **Phiên CPU (Không tốn GPU quota):**
  Dùng để tune các mô hình chạy CPU hiệu quả: `RandomForestClassifier` và `LogisticRegression`.
  ```bash
  !git clone https://github.com/gnUrt1106/Fraud-XAI.git
  %cd Fraud-XAI
  !pip install -r requirements.txt
  !python tune_hyperparams.py --model cpu --trials 30 --patience 5
  ```
  *(RandomForest sẽ tự động sử dụng song song tất cả các nhân CPU có sẵn trên Kaggle).*

---

## 3. Chạy Thí nghiệm

Sau khi nhận được các tham số tối ưu (được cập nhật tự động hoặc thủ công trong `configs/default.yaml` và `src/models.py`), bạn hãy chạy script thí nghiệm chính để tạo toàn bộ dữ liệu phân tích.

```bash
# Chạy toàn bộ thí nghiệm (4 mô hình × 3 điều kiện mất cân bằng)
python run_experiments.py
```

* **Lưu ý:** 
  - Kết quả metrics chi tiết được lưu trữ trong `outputs/results/` dưới dạng file JSON/CSV.

---

## 4. Trực quan hóa trên Streamlit Dashboard (Local)

Mở terminal tại máy local, kích hoạt venv và khởi chạy ứng dụng Streamlit:

```bash
streamlit run dashboard/app.py
```

Ứng dụng sẽ tự động mở tại **http://localhost:8501** gồm các tab chính:
- **Tab 1 — Performance Metrics:** So sánh PR-AUC, F1-Score, Recall giữa các điều kiện cân bằng dữ liệu khác nhau dưới dạng biểu đồ cột trực quan.
- **Tab 2 — Detailed Analysis (Heatmap & Scatter):** Phân tích chi tiết dưới dạng Heatmap theo cặp Mô hình × Điều kiện, và biểu đồ Scatter phân tích sự đánh đổi (trade-off) giữa PR-AUC và F1.
- **Tab 3 — Full Results Table:** Bảng tổng hợp đầy đủ kết quả chỉ số kèm theo tính năng tải về (Download) file CSV.
