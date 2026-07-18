"""
ANN (PyTorch) model definition + Optuna search space.

Nhánh Deep Learning — song song với nhánh ML (models_ml.py).
Cùng sử dụng fold indices từ cv.py, cùng imbalance handling,
cùng evaluation metric (PR-AUC) để so sánh paired được công bằng.

Kiến trúc mặc định:
    Input(30)
      → Dense(64) → BatchNorm → ReLU → Dropout(0.3)
      → Dense(32) → BatchNorm → ReLU → Dropout(0.2)
      → Dense(16) → ReLU
      → Dense(1)  → Sigmoid
"""

import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ── Device ────────────────────────────────────────────────────────────

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()  # Apple M2 Pro
    else "cpu"
)
logger.info("ANN device: %s", DEVICE)


# ── Model Definition ──────────────────────────────────────────────────

class FraudANN(nn.Module):
    """
    Fully-connected ANN for binary fraud detection.

    Hỗ trợ n_layers và units linh hoạt (Optuna search).
    BatchNorm sau mỗi Dense giúp ổn định training với imbalanced data.
    Dropout chống overfit — đặc biệt quan trọng khi dùng SMOTE oversampling.
    """

    def __init__(
        self,
        input_dim: int = 30,
        n_layers: int = 3,
        units: list[int] = None,
        dropout_rates: list[float] = None,
    ):
        """
        Args:
            input_dim:    Số features đầu vào.
            n_layers:     Số hidden layers.
            units:        List số neurons mỗi layer.
                          Mặc định: [64, 32, 16] (3 layers).
            dropout_rates: List dropout rate mỗi layer.
                           Mặc định: [0.3, 0.2, 0.0].
        """
        super().__init__()
        units = units or [64, 32, 16]
        dropout_rates = dropout_rates or [0.3, 0.2, 0.0]

        # Pad/truncate nếu n_layers khác độ dài list
        units = (units * n_layers)[:n_layers]
        dropout_rates = (dropout_rates * n_layers)[:n_layers]

        layers = []
        prev_dim = input_dim
        for i in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, units[i]),
                nn.BatchNorm1d(units[i]),
                nn.ReLU(),
            ])
            if dropout_rates[i] > 0:
                layers.append(nn.Dropout(dropout_rates[i]))
            prev_dim = units[i]

        layers.append(nn.Linear(prev_dim, 1))
        # Không dùng Sigmoid ở đây vì BCEWithLogitsLoss đã tích hợp sigmoid
        # (numerically stable hơn). Khi inference thì wrap qua sigmoid.
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Inference: trả về xác suất fraud (dùng cho sklearn-compatible evaluation).

        Args:
            X: numpy array shape (n_samples, n_features).

        Returns:
            proba: numpy array shape (n_samples,), giá trị trong [0, 1].
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            logits = self.forward(X_tensor)
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba


# ── Training ──────────────────────────────────────────────────────────

def train_ann(
    model: FraudANN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 10,
    pos_weight: Optional[float] = None,
    seed: int = 42,
) -> dict:
    """
    Train ANN với EarlyStopping theo val PR-AUC.

    Args:
        model:         FraudANN instance (chưa fit).
        X_train, y_train: Training data (numpy).
        X_val, y_val:     Validation data (numpy).
        learning_rate: Adam LR.
        batch_size:    Mini-batch size.
        max_epochs:    Giới hạn epoch tối đa.
        patience:      Số epoch không cải thiện val PR-AUC trước khi dừng.
        pos_weight:    Weight cho positive class trong BCEWithLogitsLoss.
                       Truyền vào khi dùng class-weighting.
                       (= n_negative / n_positive, ví dụ ≈ 578 với dataset này)
        seed:          Seed cho DataLoader worker.

    Returns:
        Dict với:
            - 'best_val_prauc': float
            - 'epochs_trained': int
            - 'history': list of (train_loss, val_prauc) per epoch
    """
    from sklearn.metrics import average_precision_score

    torch.manual_seed(seed)

    model = model.to(DEVICE)

    # Loss
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoader
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_val_prauc = -1.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(max_epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        avg_loss = total_loss / len(X_train)

        # ── Validate ──
        val_proba = model.predict_proba(X_val)
        val_prauc = average_precision_score(y_val, val_proba)

        history.append((avg_loss, val_prauc))

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == patience:
            logger.debug(
                "Epoch %3d | loss=%.4f | val_prauc=%.4f | patience=%d",
                epoch, avg_loss, val_prauc, patience_counter,
            )

        if patience_counter >= patience:
            logger.info(
                "EarlyStopping at epoch %d (best val PR-AUC=%.4f)",
                epoch, best_val_prauc,
            )
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_prauc": best_val_prauc,
        "epochs_trained": epoch + 1,
        "history": history,
    }


# ── Optuna Objective ──────────────────────────────────────────────────

def optuna_objective_ann(
    trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    use_pos_weight: bool = False,
    seed: int = 42,
) -> float:
    """
    Optuna objective function cho ANN — tối ưu PR-AUC trên validation fold.

    Search space (theo spec):
        n_layers:      2–4
        units_l0:      {32, 64, 128}
        dropout:       0.1–0.5
        learning_rate: 1e-4–1e-2 (log scale)
        batch_size:    {128, 256, 512}

    Args:
        trial:         Optuna Trial object.
        use_pos_weight: Nếu True, tính pos_weight từ y_train để dùng class-weight.

    Returns:
        val_prauc: float (để maximize).
    """
    n_layers = trial.suggest_int("n_layers", 2, 4)
    units_l0 = trial.suggest_categorical("units_l0", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    # Tạo units list theo n_layers: giảm dần kích thước
    units = [max(16, units_l0 // (2 ** i)) for i in range(n_layers)]
    dropout_rates = [dropout] * n_layers

    model = FraudANN(
        input_dim=input_dim,
        n_layers=n_layers,
        units=units,
        dropout_rates=dropout_rates,
    )

    pos_weight = None
    if use_pos_weight:
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = float(n_neg / max(n_pos, 1))

    result = train_ann(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        learning_rate=lr,
        batch_size=batch_size,
        pos_weight=pos_weight,
        seed=seed,
    )
    return result["best_val_prauc"]
