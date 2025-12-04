import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd

def create_advanced_features(df):
    df = df.copy()

    # ========== 基础要求：确保有 ds ========== #
    if "ds" not in df.columns:
        raise ValueError("DataFrame 必须包含 'ds' 列作为时间戳。")

    df["ds"] = pd.to_datetime(df["ds"])

    # ========== 1. 日期时间特征 ========== #
    df["hour"] = df["ds"].dt.hour
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["dayofyear"] = df["ds"].dt.dayofyear

    # 周期编码
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # ========== 2. 自定义季节特征 ========== #
    def get_season(m):
        if m in [12, 1, 2]:
            return 0  # Winter
        elif m in [3, 4, 5]:
            return 1  # Spring
        elif m in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn

    df["season"] = df["month"].apply(get_season)

    # One-hot season
    df = pd.get_dummies(df, columns=["season"], prefix="season")

    # ========== 3. 风向特征（sin/cos） ========== #
    if "wind_direction" in df.columns:
        df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction"]))
        df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction"]))
    else:
        df["wind_dir_sin"] = 0
        df["wind_dir_cos"] = 0

    # ========== 4. rolling 平滑（不限于 wind_speed） ========== #
    rolling_cols = []
    if "wind_speed" in df.columns:
        rolling_cols.append("wind_speed")
    if "wind_gust" in df.columns:
        rolling_cols.append("wind_gust")
    if "temperature" in df.columns:
        rolling_cols.append("temperature")

    for col in rolling_cols:
        df[f"{col}_roll3"] = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_roll6"] = df[col].rolling(6, min_periods=1).mean()
        df[f"{col}_roll12"] = df[col].rolling(12, min_periods=1).mean()

    # ========== 5. lag 特征（避免泄漏） ========== #
    lag_features = []
    if "y" in df.columns:
        lag_features.append("y")
    if "wind_speed" in df.columns:
        lag_features.append("wind_speed")

    for col in lag_features:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag24"] = df[col].shift(24)
        df[f"{col}_lag48"] = df[col].shift(48)

    # ========== 6. 缺失值处理 ========== #
    df = df.fillna(method="ffill").fillna(method="bfill")
    # 删除无关列
    df = df.drop(['hourly__time', 'weekday', 'season'], axis=1, errors='ignore')

    return df


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class LSTM:
    """
    干净版：不做特征工程
    你需要在外部先做好 create_advanced_features()
    """

    def __init__(self, seq_len=24, hidden_dim=128, num_layers=2, dropout=0.2, device=None):
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 数据处理器
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        # 保存训练时的特征列顺序
        self.feature_cols = None

        # Pytorch 模型
        self.model = None
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    # ---------------------- Data Utils ---------------------- #

    def _create_sequences(self, X, y):
        seq_len = self.seq_len
        X_seq, y_seq = [], []

        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])

        return np.array(X_seq), np.array(y_seq)

    # ---------------------- Fit ---------------------- #

    def fit(self, df, target_col="y", epochs=10, lr=1e-3, batch=128):

        # 1. 记录特征列（除 y 以外）
        self.feature_cols = [c for c in df.columns if c != target_col]

        # 2. 分离 X, y
        X = df[self.feature_cols].values
        y = df[target_col].values.reshape(-1, 1)

        # Remove datetime columns
        datetime_cols = X.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
        if len(datetime_cols) > 0:
            print(f"[Warning] Dropping datetime columns: {list(datetime_cols)}")
            X = X.drop(columns=datetime_cols)

        # 3. Impute + scale
        X = self.imputer.fit_transform(X)
        X = self.scaler_x.fit_transform(X)
        y = self.scaler_y.fit_transform(y)

        # 4. 构建序列
        X_seq, y_seq = self._create_sequences(X, y)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(self.device)

        # 5. 初始化模型
        self.model = LSTMRegressor(
            input_dim=X.shape[1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # 6. 训练循环
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for ep in range(epochs):
            self.model.train()
            idx = np.random.permutation(len(X_tensor))

            for i in range(0, len(idx), batch):
                batch_idx = idx[i:i+batch]
                xb = X_tensor[batch_idx]
                yb = y_tensor[batch_idx]

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                optimizer.step()

            print(f"Epoch {ep+1}/{epochs} - Loss={loss.item():.6f}")

    # ---------------------- Predict ---------------------- #

    def predict(self, df_future, target_col="y"):
        """
        注意：df_future 必须保留 ds 列，但不能包含 y。
        """

        if "ds" not in df_future.columns:
            raise ValueError("predict 时必须包含 ds 列以输出 forecast DataFrame。")

        # 必须使用训练时的特征列顺序
        X = df_future[self.feature_cols].values

        # 相同的 impute + scale
        X = self.imputer.transform(X)
        X = self.scaler_x.transform(X)

        # 构建输入序列（只适用于单步预测）
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i in range(self.seq_len, len(X_tensor)):
                seq = X_tensor[i-self.seq_len:i].unsqueeze(0)   # shape (1, seq_len, n_features)
                y_pred = self.model(seq)
                y_pred = y_pred.cpu().numpy()
                y_pred = self.scaler_y.inverse_transform(y_pred)
                preds.append(y_pred[0][0])

        # 对齐 ds（前 seq_len 无预测）
        ds_out = df_future["ds"].iloc[self.seq_len:].reset_index(drop=True)

        return pd.DataFrame({
            "ds": ds_out,
            "yhat": preds
        })

