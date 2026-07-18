# Implementation spec — Fraud detection: nhánh Machine Learning vs Deep Learning

## Bối cảnh

Dự án hiện có: `~/Projects/anomaly-detection-framework/`, dataset ULB Credit Card Fraud (`creditcard.csv`: `Time`, `V1`-`V28`, `Amount`, `Class`, 284807 dòng, 492 fraud).

Đã có sẵn ma trận thực nghiệm 4 model cây × 3 điều kiện imbalance (12 config): XGBoost, CatBoost, RandomForest, LogisticRegression × {class-weight, SMOTE, SMOTE-ENN}, tune bằng Optuna.

Task: thêm nhánh Deep Learning (ANN) song song, chuẩn hoá lại cross-validation và evaluation để 2 nhánh so sánh được công bằng, và dùng SHAP làm phương pháp XAI duy nhất (đã bỏ CIES, LIME, Integrated Gradients).

## 1. Cấu trúc thư mục đề xuất

```
anomaly-detection-framework/
  data/
    creditcard.csv
  src/
    data_profiling.py     # thống kê mô tả dữ liệu, xuất HTML report
    preprocessing.py      # scale Amount/Time, load data
    cv.py                 # StratifiedKFold wrapper, fixed seeds
    imbalance.py           # SMOTE / SMOTE-ENN / class-weight per branch
    models_ml.py           # XGBoost, CatBoost, RF, LogReg + Optuna objectives
    models_dl.py            # ANN (PyTorch), Optuna objective
    evaluation.py           # threshold selection, metrics, paired tests
    explain.py               # SHAP: TreeExplainer / DeepExplainer
    run_experiment.py         # orchestrates toàn bộ pipeline
  configs/
    experiment.yaml
  outputs/
    profiling/
    metrics/
    shap_values/
    models/
```

## 2. Thống kê mô tả dữ liệu (`data_profiling.py`)

Chạy **một lần duy nhất**, trên toàn bộ `creditcard.csv` gốc, **trước khi chia fold** — đây thuần là bước mô tả dữ liệu (EDA), không dùng để train hay tune nên không có vấn đề leakage.

- Dùng thư viện `ydata-profiling` (tên cũ `pandas-profiling`).
- Output: `outputs/profiling/data_profile.html`.

```python
from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("data/creditcard.csv")
profile = ProfileReport(
    df,
    title="ULB Credit Card Fraud — data profiling",
    explorative=True,
)
profile.to_file("outputs/profiling/data_profile.html")
```

Report này nên bao gồm (mặc định `ydata-profiling` đã tự làm):
- Phân phối, missing values, cardinality của từng biến `V1`-`V28`, `Amount`, `Time`.
- Tỉ lệ mất cân bằng của `Class` (492/284807 ≈ 0.17%) — nêu rõ ngay đầu report để làm căn cứ cho phần chọn imbalance handling ở bước 4-5.
- Ma trận tương quan giữa các biến — vì `V1`-`V28` đã qua PCA nên tương quan giữa chúng phải gần 0; nếu report cho thấy tương quan cao bất thường, đó là dấu hiệu cần kiểm tra lại dữ liệu trước khi train.
- Phân phối `Amount` tách riêng theo `Class` (fraud vs không fraud) — hữu ích để giải thích lý do cần scale `Amount` trước khi vào ANN.

Lưu ý: `ydata-profiling` khá nặng với ~284k dòng × 31 cột — nếu chạy chậm, bật `minimal=True` trong `ProfileReport(...)` để bỏ bớt phần tương tác/tính toán nặng (pairwise correlations chi tiết).

## 3. Preprocessing (`preprocessing.py`)

- Load `creditcard.csv`.
- Tách `X` (30 cột: `Time`, `V1`-`V28`, `Amount`) và `y` (`Class`).
- **Không fit scaler trên toàn bộ dữ liệu.** Scaler (`StandardScaler` hoặc `RobustScaler`) chỉ fit trên phần train của từng fold, rồi transform cả train lẫn validation của fold đó — tránh leakage.
- Chỉ cần scale `Amount` và `Time`; `V1`-`V28` đã là output PCA, không cần xử lý thêm.

## 4. Cross-validation (`cv.py`)

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`.
- `SEED` cố định một chỗ, dùng lại cho: numpy, `random`, sklearn, và framework DL (torch/tf) — set ở đầu `run_experiment.py`, không set riêng lẻ trong từng file.
- Cùng 1 bộ fold indices được dùng cho **cả 2 nhánh** — bắt buộc, để so sánh paired được ở bước 6.

## 5. Nhánh Machine Learning (`models_ml.py`)

Giữ nguyên logic hiện có: với mỗi fold, với mỗi điều kiện imbalance trong {class-weight, SMOTE, SMOTE-ENN}:
- Áp dụng imbalance handling **chỉ trên tập train của fold** (SMOTE/SMOTE-ENN không bao giờ chạm vào validation).
- Optuna objective tối ưu theo PR-AUC trên validation fold.
- 4 model: XGBoost, CatBoost, RandomForest, LogisticRegression.

## 6. Nhánh Deep Learning (`models_dl.py`)

### Kiến trúc ANN
```
Input(30)
  -> Dense(64) -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Dense(32) -> BatchNorm -> ReLU -> Dropout(0.2)
  -> Dense(16) -> ReLU
  -> Dense(1) -> Sigmoid
```

### Xử lý mất cân bằng cho ANN
- **class-weight**: weighted BCE loss (`pos_weight` trong `BCEWithLogitsLoss`).
- **SMOTE / SMOTE-ENN**: áp dụng trên train fold trước khi đưa vào DataLoader, giống hệt cách nhánh ML làm — để 2 nhánh dùng đúng cùng 1 dữ liệu train đã qua imbalance handling khi so sánh.
- Cân nhắc `WeightedRandomSampler` như một biến thể bổ sung nếu oversampling thô gây overfit (ghi chú lại, không bắt buộc).

### Optuna search space
```python
{
  "n_layers": trial.suggest_int("n_layers", 2, 4),
  "units_l0": trial.suggest_categorical("units_l0", [32, 64, 128]),
  "dropout": trial.suggest_float("dropout", 0.1, 0.5),
  "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
  "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
}
```
- Optimize theo PR-AUC trên validation fold — cùng metric objective với nhánh ML.
- `EarlyStopping` theo dõi val PR-AUC (không dùng accuracy — vô nghĩa với imbalance nặng như thế này).
- Train trên Kaggle GPU khi tune (nhiều trial), infer/test lại trên local M2 Pro nếu cần.

## 7. Evaluation chuẩn hoá giữa 2 nhánh (`evaluation.py`)

Đây là phần bắt buộc phải thống nhất, không được để mỗi nhánh tự làm khác nhau:

1. **Threshold rule cố định cho cả 5 model**: threshold tối đa hoá F1 trên tập validation của từng fold (không dùng 0.5 mặc định, không để mỗi model tự chọn threshold "có lợi").
2. **Report mean ± std qua 5 fold** cho từng metric (PR-AUC, Recall, Precision), không chỉ báo cáo 1 số gộp.
3. **Paired comparison**: vì cả 2 nhánh dùng chung fold indices, chạy Wilcoxon signed-rank test (hoặc paired t-test nếu phân phối phù hợp) so từng cặp model trên từng fold tương ứng — để kết luận "model A tốt hơn B" có ý nghĩa thống kê, không chỉ so trung bình.
4. **Seed cố định** xuyên suốt (đã nói ở bước 3) để loại trừ variance do random init của ANN.

## 8. Explainability (`explain.py`)

- Model cây (XGBoost, CatBoost, RF): `shap.TreeExplainer` — exact Shapley values.
- LogisticRegression: `shap.LinearExplainer`.
- ANN: `shap.DeepExplainer` hoặc `shap.GradientExplainer` — approximate, cần chọn background sample cố định (ví dụ 100 sample từ train fold, seed cố định) để kết quả reproducible.
- **Ghi chú bắt buộc trong report**: giá trị SHAP giữa nhánh cây (exact) và nhánh ANN (approximate) không cùng độ tin cậy — không so sánh giá trị SHAP tuyệt đối giữa 2 nhánh như thể chúng cùng thang đo.
- Output: lưu SHAP values mỗi fold/model vào `outputs/shap_values/{model}_{condition}_fold{k}.npy` hoặc tương tự, kèm summary plot (mean |SHAP|) per model.

## 9. Orchestration (`run_experiment.py`)

- Set global seed đầu tiên.
- Load config từ `configs/experiment.yaml` (danh sách model, imbalance conditions, K folds, Optuna n_trials).
- Loop: fold → nhánh (ML/DL) → imbalance condition → model → Optuna tune → train final → evaluate → SHAP → save.
- Tổng: (4 model ML + 1 ANN) × 3 điều kiện imbalance × 5 fold = 75 runs.
- Output cuối: 1 bảng tổng hợp (`outputs/metrics/summary.csv`) với cột: model, condition, fold, PR-AUC, Recall, Precision, threshold; và 1 bảng kết quả paired test.

## 10. Definition of done

- [ ] Cùng 1 bộ StratifiedKFold indices dùng cho cả ML và DL.
- [ ] Imbalance handling chạy trong từng fold, không leakage.
- [ ] Threshold chọn theo rule thống nhất (F1-optimal per fold), áp cho cả 5 model.
- [ ] Seed cố định xuyên suốt, log lại giá trị seed đã dùng.
- [ ] SHAP chạy được cho cả 5 model, có ghi chú giới hạn approximate của DeepExplainer.
- [ ] `summary.csv` có mean ± std qua fold + kết quả paired test giữa các model.
- [ ] `outputs/profiling/data_profile.html` được tạo từ dữ liệu gốc trước khi chia fold.
