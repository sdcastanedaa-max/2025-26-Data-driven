import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class RandomForestModel:
    """
    Random Forest 风电预测模型
    """

    def __init__(self, **kwargs):
        default_params = {
            'n_estimators': 300,
            'max_depth': 20,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = RandomForestRegressor(**default_params)
        self.is_fitted = False


    # -------------------------
    #  天气字符串分类
    # -------------------------
    def _categorize_weather(self, w):
        if pd.isna(w):
            return (0, 0, 0)

        w = w.lower()

        # 云量 CloudLevel
        if "clear" in w:
            cloud = 0
        elif "partly" in w:
            cloud = 1
        elif "cloudy" in w:
            cloud = 2
        elif "overcast" in w:
            cloud = 3
        else:
            cloud = 1

        # 雨强度 RainLevel
        if "drizzle" in w or "slight" in w:
            rain = 1
        elif "rain" in w:
            rain = 2
        elif "heavy" in w:
            rain = 3
        else:
            rain = 0

        # 风暴 StormLevel
        if "storm" in w or "thunder" in w:
            storm = 1
        else:
            storm = 0

        return (cloud, rain, storm)


    # -------------------------
    #  特征工程
    # -------------------------
    def _create_features(self, df):
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])

        # --- 时间周期特征 ---
        df["hour"] = df["ds"].dt.hour
        df["month"] = df["ds"].dt.month

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # --- 风向 ---
        if "hourly__wind_direction_100m" in df.columns:
            rad = np.deg2rad(df["hourly__wind_direction_100m"])
            df["wind_dir_sin"] = np.sin(rad)
            df["wind_dir_cos"] = np.cos(rad)

        # --- 天气分类 ---
        if "daily__weather_code" in df.columns:
            parsed = df["daily__weather_code"].apply(self._categorize_weather)
            df["cloud_level"] = [p[0] for p in parsed]
            df["rain_level"] = [p[1] for p in parsed]
            df["storm_level"] = [p[2] for p in parsed]
        else:
            df["cloud_level"] = 0
            df["rain_level"] = 0
            df["storm_level"] = 0

        # --- 季节因子 ---
        if "season_factor" not in df.columns:
            df["season_factor"] = 0.6

        # --- 保留核心特征 ---
        keep = [
            "hour", "month",
            "hour_sin", "hour_cos",
            "month_sin", "month_cos",
            "hourly__wind_speed_100m",
            "hourly__apparent_temperature",
            "wind_dir_sin", "wind_dir_cos",
            "cloud_level", "rain_level", "storm_level",
            "season_factor"
        ]

        return df[[c for c in keep if c in df.columns]]


    # -------------------------
    #  训练模型
    # -------------------------
    def fit(self, df):
        features = self._create_features(df)
        target = df["y"]

        self.feature_columns = features.columns.tolist()
        self.model.fit(features, target)
        self.is_fitted = True
        return self


    # -------------------------
    #  预测
    # -------------------------
    def predict(self, df):
        if not self.is_fitted:
            raise ValueError("先调用 fit()")

        features = self._create_features(df)
        y_pred = self.model.predict(features)

        return pd.DataFrame({
            "ds": df["ds"],
            "yhat": y_pred
        })
