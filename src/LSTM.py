import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')


# =============================
#  独立特征工程函数
# =============================
def create_advanced_features(df):
    df = df.copy()

    # 关键修复：训练集有 y，但预测集没有；避免 y 被当作特征
    if 'y' in df.columns:
        df = df.drop(columns=['y'], errors='ignore')

    # 1. 时间列
    if 'ds' not in df.columns:
        raise ValueError("输入数据必须包含 'ds' 列（datetime）")
    df['ds'] = pd.to_datetime(df['ds'])
    # df = df.sort_values('ds').reset_index(drop=True)

    # 2. weekday, is_weekend, season
    if 'weekday' in df.columns:
        if df['weekday'].dtype == 'object':
            df['weekday_index'] = df['weekday'].astype('category').cat.codes
        else:
            df['weekday_index'] = df['weekday']

    if 'is_weekend' in df.columns:
        if df['is_weekend'].dtype == 'bool':
            df['is_weekend_flag'] = df['is_weekend'].astype(int)
        else:
            df['is_weekend_flag'] = df['is_weekend'].astype('category').cat.codes

    if 'season' in df.columns:
        if df['season'].dtype == 'object':
            df['season_index'] = df['season'].astype('category').cat.codes
        else:
            df['season_index'] = df['season']

    # 3. 基本时间特征
    df['hour'] = df['ds'].dt.hour
    df['month'] = df['ds'].dt.month
    df['day_of_year'] = df['ds'].dt.dayofyear

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # 4. 风向
    if 'hourly__wind_direction_100m' in df.columns:
        angles = pd.to_numeric(df['hourly__wind_direction_100m'], errors='coerce')
        radians = np.deg2rad(angles.fillna(0.0))
        df['wind_direction_sin'] = np.sin(radians)
        df['wind_direction_cos'] = np.cos(radians)

    # 5. 风速
    if 'hourly__wind_speed_100m' in df.columns:
        ws = pd.to_numeric(df['hourly__wind_speed_100m'], errors='coerce')
        df['ws'] = ws
        df['ws_squared'] = ws**2
        df['ws_cubed'] = ws**3
        df['ws_log'] = np.log1p(ws.clip(lower=0))

        for lag in [1, 2, 3, 6]:
            df[f'ws_lag_{lag}'] = ws.shift(lag)

        for window in [3, 6]:
            df[f'ws_roll_mean_{window}'] = ws.rolling(window, min_periods=1).mean()
            df[f'ws_roll_std_{window}'] = ws.rolling(window, min_periods=1).std()

        df['ws_diff_1'] = ws.diff(1)
        df['ws_diff_3'] = ws.diff(3)

    # 6. 风向滞后
    if 'wind_direction_sin' in df.columns:
        for lag in [1, 3]:
            df[f'wind_sin_lag_{lag}'] = df['wind_direction_sin'].shift(lag)
            df[f'wind_cos_lag_{lag}'] = df['wind_direction_cos'].shift(lag)

    # 7. 温度
    if 'hourly__apparent_temperature' in df.columns:
        temp = pd.to_numeric(df['hourly__apparent_temperature'], errors='coerce')
        df['temp'] = temp
        df['temp_lag_1'] = temp.shift(1)
        df['temp_roll_mean_3'] = temp.rolling(3, min_periods=1).mean()
        if 'ws' in df.columns:
            df['temp_ws_interaction'] = temp * df['ws']

    # 8. 删除无关列
    df = df.drop(['ds', 'hourly__time', 'weekday', 'season'], axis=1, errors='ignore')

    # 9. 只保留数值
    df = df.select_dtypes(include=[np.number])

    return df


# =============================
# 内部 LSTM 模型
# =============================
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # 取序列最后一步
        out = self.fc(out)
        return out


# =============================
# 时间序列 Dataset
# =============================
class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len=24):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        return x_seq, y_val


# =============================
# 主模型（保持接口不变）
# =============================
class LSTM:
    """
    PyTorch LSTM 版本（接口与原 LightGBM 完全兼容）
    """

    def __init__(self, seq_len=24, hidden_dim=128, num_layers=2, dropout=0.2, **kwargs):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

        self.is_fitted = False
        self.feature_names = None

        self.y_scaler = StandardScaler()

        df = df.copy()

        # 关键修复：训练集有 y，但预测集没有；避免 y 被当作特征
        if 'y' in df.columns:
            df = df.drop(columns=['y'], errors='ignore')

        # 1. 统一 datetime
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
        else:
            raise ValueError("输入数据必须包含 'ds' 列（datetime）。")

        df = df.sort_values('ds').reset_index(drop=True)

        # 2. 处理常见字符串/布尔列： weekday, is_weekend, season
        if 'weekday' in df.columns:
            if df['weekday'].dtype == 'object' or pd.api.types.is_categorical_dtype(df['weekday']):
                df['weekday_index'] = df['weekday'].astype('category').cat.codes
            else:
                df['weekday_index'] = df['weekday']

        if 'is_weekend' in df.columns:
            if df['is_weekend'].dtype == 'bool':
                df['is_weekend_flag'] = df['is_weekend'].astype(int)
            elif df['is_weekend'].dtype == 'object':
                df['is_weekend_flag'] = df['is_weekend'].map({
                    'True': 1, 'true': 1, 'TRUE': 1,
                    'False': 0, 'false': 0, 'FALSE': 0
                }).fillna(df['is_weekend'].astype('category').cat.codes)
            else:
                df['is_weekend_flag'] = df['is_weekend'].astype(int)

        if 'season' in df.columns:
            if df['season'].dtype == 'object' or pd.api.types.is_categorical_dtype(df['season']):
                df['season_index'] = df['season'].astype('category').cat.codes
            else:
                df['season_index'] = df['season']

        # 3. 基本时间特征
        df['hour'] = df['ds'].dt.hour
        df['month'] = df['ds'].dt.month
        df['day_of_year'] = df['ds'].dt.dayofyear

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        # 4. 风向
        if 'wind_direction_sin' not in df.columns or 'wind_direction_cos' not in df.columns:
            if 'hourly__wind_direction_100m' in df.columns:
                angles = pd.to_numeric(df['hourly__wind_direction_100m'], errors='coerce')
                radians = np.deg2rad(angles.fillna(0.0))
                df['wind_direction_sin'] = np.sin(radians)
                df['wind_direction_cos'] = np.cos(radians)

        # 5. 风速
        if 'hourly__wind_speed_100m' in df.columns:
            ws = pd.to_numeric(df['hourly__wind_speed_100m'], errors='coerce')
            df['ws'] = ws
            df['ws_squared'] = ws ** 2
            df['ws_cubed'] = ws ** 3
            df['ws_log'] = np.log1p(ws.clip(lower=0))

            for lag in [1, 2, 3, 6]:
                df[f'ws_lag_{lag}'] = ws.shift(lag)

            for window in [3, 6]:
                df[f'ws_roll_mean_{window}'] = ws.rolling(window, min_periods=1).mean()
                df[f'ws_roll_std_{window}'] = ws.rolling(window, min_periods=1).std()

            df['ws_diff_1'] = ws.diff(1)
            df['ws_diff_3'] = ws.diff(3)

        # 6. 风向滞后
        for lag in [1, 3]:
            if 'wind_direction_sin' in df.columns:
                df[f'wind_sin_lag_{lag}'] = pd.to_numeric(df['wind_direction_sin'], errors='coerce').shift(lag)
            if 'wind_direction_cos' in df.columns:
                df[f'wind_cos_lag_{lag}'] = pd.to_numeric(df['wind_direction_cos'], errors='coerce').shift(lag)

        # 7. 温度
        if 'hourly__apparent_temperature' in df.columns:
            temp = pd.to_numeric(df['hourly__apparent_temperature'], errors='coerce')
            df['temp'] = temp
            df['temp_lag_1'] = temp.shift(1)
            df['temp_roll_mean_3'] = temp.rolling(3, min_periods=1).mean()
            if 'ws' in df.columns:
                df['temp_ws_interaction'] = temp * df['ws']

        # 8. 天气编码
        if 'daily__weather_code' in df.columns:
            parsed = df['daily__weather_code'].apply(self._categorize_weather)
            df['cloud_level'] = parsed.apply(lambda x: x[0])
            df['rain_level'] = parsed.apply(lambda x: x[1])
            df['storm_level'] = parsed.apply(lambda x: x[2])
        else:
            df['cloud_level'] = 1
            df['rain_level'] = 0
            df['storm_level'] = 0

        # 9. 其他文本/布尔 → 数值
        for col in df.columns:
            if col in ['daily__weather_code', 'hourly__time', 'ds']:
                continue
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)
            elif df[col].dtype == 'object':
                df[col] = df[col].astype('category').cat.codes

        # 10. 删除无用列
        df = df.drop(['ds', 'hourly__time', 'weekday', 'season'], axis=1, errors='ignore')

        # 只保留数值
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        return numeric_df
    # ------------------------------------------------------------------
    # 天气分类
    # ------------------------------------------------------------------
    def _categorize_weather(self, w):
        if pd.isna(w):
            return (1, 0, 0)

        w = str(w).lower()

        if 'clear' in w:
            cloud = 0
        elif 'partly' in w:
            cloud = 1
        elif 'cloud' in w:
            cloud = 2
        elif 'overcast' in w:
            cloud = 3
        else:
            cloud = 1

        if 'heavy' in w:
            rain = 3
        elif 'rain' in w:
            rain = 2
        elif 'drizzle' in w:
            rain = 1
        else:
            rain = 0

        storm = 1 if ('storm' in w or 'thunder' in w) else 0

        return cloud, rain, storm

    def _get_sample_weights(self, df):
        hours = pd.to_datetime(df['ds']).dt.hour
        weights = np.ones(len(df))
        weights[(hours >= 10) & (hours <= 20)] = 1.0
        weights[(hours >= 14) & (hours <= 16)] = 1.0
        return weights

    # -----------------------------------------
    # 训练
    # -----------------------------------------
    def fit(self, df, eval_df=None, epochs=5, batch_size=64, lr=1e-3):
        print("创建特征...")
        X = df.drop(columns=['y'], errors='ignore').values.astype(np.float32)
        y = self.y_scaler.fit_transform(y.reshape(-1,1)).flatten().astype(np.float32)

        print("缺失值处理...")
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)

        X = X.astype(np.float32)

        # 构造 DataLoader
        train_ds = SeqDataset(X, y, seq_len=self.seq_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        input_dim = X.shape[1]
        print("初始化 LSTM...")
        self.model = LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.L1Loss()

        print("开始训练 LSTM...")
        for ep in range(epochs):
            epoch_loss = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device).unsqueeze(1)

                pred = self.model(xb)
                loss = criterion(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {ep+1}/{epochs} - Loss: {epoch_loss:.4f}")

        self.is_fitted = True
        print("训练完成！")
        return self

    # -----------------------------------------
    # 预测（逐步预测，不泄漏未来）
    # -----------------------------------------

    def predict(self, df):
        if not self.is_fitted:
            raise ValueError("模型未训练")

        features = df.values.astype(np.float32)
        features = self.imputer.transform(features)
        features = self.scaler.transform(features)
        features = features.astype(np.float32)

        preds = []
        seq = []

        # 滚动预测
        for t in range(len(features)):
            seq.append(features[t])
            if len(seq) < self.seq_len:
                preds.append(np.nan)
                continue

            x = np.array(seq[-self.seq_len:])[None, :, :]
            x = torch.tensor(x, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                yhat = self.model(x).cpu().numpy().item()
                yhat = self.y_scaler.inverse_transform([[yhat]])[0,0]

            preds.append(yhat)

        return pd.DataFrame({
            'ds': df['ds'],
            'yhat': preds
        })

    # -----------------------------------------
    # LSTM 没有原生特征重要性 → 用权重范数代替
    # -----------------------------------------
    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("模型未训练")

        weight = self.model.lstm.weight_ih_l0.detach().cpu().numpy()
        importance = np.mean(np.abs(weight), axis=0)

        return pd.DataFrame({
            "feature": [f"f_{i}" for i in range(len(importance))],
            "importance": importance
        }).sort_values("importance", ascending=False)
