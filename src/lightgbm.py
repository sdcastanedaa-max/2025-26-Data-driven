import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class LightGBM:
    """
    修复版：兼容 train_df 的 LightGBM 风电预测模型
    """

def __init__(self, **kwargs):
    default_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
        'min_data_in_leaf': 20,
        'max_depth': -1
    }
    default_params.update(kwargs)

    self.model = None
    self.params = default_params
    self.imputer = SimpleImputer(strategy='median')
    self.scaler = StandardScaler()
    self.is_fitted = False
    self.feature_names = None


# ------------------------------------------------------------------
# 修复的特征工程
# ------------------------------------------------------------------
def _create_advanced_features(self, df):
    df = df.copy()

    # 1. 统一 datetime
    df["ds"] = pd.to_datetime(df["ds"])

    df = df.sort_values("ds").reset_index(drop=True)

    # 2. weekday（字符串→数字编码）
    if df["weekday"].dtype == "object":
        df["weekday_index"] = df["weekday"].astype("category").cat.codes
    else:
        df["weekday_index"] = df["weekday"]

    # 3. season（字符串→数字编码）
    if df["season"].dtype == "object":
        df["season_index"] = df["season"].astype("category").cat.codes
    else:
        df["season_index"] = df["season"]

    # 4. 时间特征
    df["hour"] = df["ds"].dt.hour
    df["day_of_year"] = df["ds"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # 5. 风速特征
    if "hourly__wind_speed_100m" in df.columns:
        ws = df["hourly__wind_speed_100m"]
        df["ws_squared"] = ws ** 2
        df["ws_cubed"] = ws ** 3
        df["ws_log"] = np.log1p(ws)

        # 滞后特征
        for lag in [1, 2, 3, 6]:
            df[f"ws_lag_{lag}"] = ws.shift(lag)

        # 滚动统计（不会引入不存在的字段）
        for window in [3, 6]:
            df[f"ws_roll_mean_{window}"] = ws.rolling(window, min_periods=1).mean()
            df[f"ws_roll_std_{window}"] = ws.rolling(window, min_periods=1).std()

        df["ws_diff_1"] = ws.diff(1)
        df["ws_diff_3"] = ws.diff(3)

    # 6. 风向（已经提前 sin/cos）
    # 添加滞后
    for lag in [1, 3]:
        df[f"wind_sin_lag_{lag}"] = df["wind_direction_sin"].shift(lag)
        df[f"wind_cos_lag_{lag}"] = df["wind_direction_cos"].shift(lag)

    # 7. 温度
    if "hourly__apparent_temperature" in df.columns:
        temp = df["hourly__apparent_temperature"]
        df["temp_lag_1"] = temp.shift(1)
        df["temp_roll_mean_3"] = temp.rolling(3, min_periods=1).mean()
        df["temp_ws_interaction"] = temp * df["hourly__wind_speed_100m"]

    # 8. 天气（字符串 → 分类）
    if "daily__weather_code" in df.columns:
        parsed = df["daily__weather_code"].apply(self._categorize_weather)
        df["cloud_level"] = parsed.apply(lambda x: x[0])
        df["rain_level"] = parsed.apply(lambda x: x[1])
        df["storm_level"] = parsed.apply(lambda x: x[2])
    else:
        df["cloud_level"] = 1
        df["rain_level"] = 0
        df["storm_level"] = 0

    # 9. 删除无法用于建模的 datetime 字段
    df = df.drop(["ds", "hourly__time", "weekday", "season"], axis=1, errors='ignore')

    return df


# ------------------------------------------------------------------
# 天气分类
# ------------------------------------------------------------------
def _categorize_weather(self, w):
    if pd.isna(w):
        return (1, 0, 0)

    w = w.lower()

    if "clear" in w:
        cloud = 0
    elif "partly" in w:
        cloud = 1
    elif "cloud" in w:
        cloud = 2
    elif "overcast" in w:
        cloud = 3
    else:
        cloud = 1

    if "heavy" in w:
        rain = 3
    elif "rain" in w:
        rain = 2
    elif "drizzle" in w:
        rain = 1
    else:
        rain = 0

    storm = 1 if ("storm" in w or "thunder" in w) else 0

    return cloud, rain, storm


# ------------------------------------------------------------------
# 样本权重
# ------------------------------------------------------------------
def _get_sample_weights(self, df):
    hours = df["ds"].dt.hour
    weights = np.ones(len(df))

    weights[(hours >= 10) & (hours <= 20)] = 3.0
    weights[(hours >= 14) & (hours <= 16)] = 5.0

    return weights


# ------------------------------------------------------------------
# 训练
# ------------------------------------------------------------------
def fit(self, df, eval_df=None, early_stopping_rounds=50):
    print("创建特征...")
    features = self._create_advanced_features(df)
    target = df["y"].values

    sample_weights = self._get_sample_weights(df)

    print("处理缺失...")
    features = self.imputer.fit_transform(features)

    print("特征标准化...")
    features = self.scaler.fit_transform(features)

    self.feature_names = [f"f_{i}" for i in range(features.shape[1])]

    lgb_train = lgb.Dataset(
        features, target,
        weight=sample_weights,
        feature_name=self.feature_names,
        free_raw_data=False
    )

    eval_sets = [lgb_train]
    eval_names = ["train"]

    if eval_df is not None:
        eval_features = self._create_advanced_features(eval_df)
        eval_features = self.imputer.transform(eval_features)
        eval_features = self.scaler.transform(eval_features)
        eval_target = eval_df["y"].values

        lgb_eval = lgb.Dataset(
            eval_features, eval_target,
            reference=lgb_train,
            feature_name=self.feature_names,
            free_raw_data=False
        )
        eval_sets.append(lgb_eval)
        eval_names.append("valid")

    print("开始训练 LightGBM...")
    self.model = lgb.train(
        self.params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=eval_sets,
        valid_names=eval_names,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=True),
            lgb.log_evaluation(100)
        ]
    )

    self.is_fitted = True
    print("训练完成!")
    return self


# ------------------------------------------------------------------
# 预测
# ------------------------------------------------------------------
def predict(self, df):
    if not self.is_fitted:
        raise ValueError("模型未训练")

    features = self._create_advanced_features(df)
    features = self.imputer.transform(features)
    features = self.scaler.transform(features)

    pred = self.model.predict(features)

    return pd.DataFrame({
        "ds": df["ds"],
        "yhat": pred
    })


# ------------------------------------------------------------------
# 特征重要性
# ------------------------------------------------------------------
def get_feature_importance(self):
    if self.model is None:
        raise ValueError("模型未训练")

    imp = pd.DataFrame({
        "feature": self.feature_names,
        "importance": self.model.feature_importance()
    }).sort_values("importance", ascending=False)
    return imp