import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


class RandomForestModel:
    """
    强化版 Random Forest 风电预测模型
    """

    def __init__(self, **kwargs):
        default_params = {
            'n_estimators': 800,
            'max_depth': None,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'max_features': 0.7,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = RandomForestRegressor(**default_params)
        self.is_fitted = False

        self.imputer = SimpleImputer(strategy="median")

    # ------------------------------------
    # 时间权重函数
    # ------------------------------------
    def _get_time_weights(self, df):
        """
        为10-20时的数据点分配更高的权重
        """
        df["ds"] = pd.to_datetime(df["ds"])
        hours = df["ds"].dt.hour

        # 创建权重数组，10-20时权重为2，其他时段权重为1
        weights = np.where((hours >= 10) & (hours <= 17), 2.0, 1.0)

        return weights

    # ------------------------------------
    # 滞后特征 + 周期特征 + 天气特征
    # ------------------------------------
    def _create_features(self, df):
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds")

        # --------------------
        # 时间
        # --------------------
        df["hour"] = df["ds"].dt.hour
        df["month"] = df["ds"].dt.month
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # --------------------
        # 风速非线性（v² v³）
        # --------------------
        if "hourly__wind_speed_100m" in df.columns:
            ws = df["hourly__wind_speed_100m"]
            df["ws2"] = ws ** 2
            df["ws3"] = ws ** 3

        # --------------------
        # 滞后特征
        # --------------------
        if "hourly__wind_speed_100m" in df.columns:
            df["ws_lag1"] = ws.shift(1)
            df["ws_lag3"] = ws.shift(3)
            df["ws_lag6"] = ws.shift(6)
            df["ws_roll3"] = ws.rolling(3).mean()
            df["ws_roll6"] = ws.rolling(6).std()

        # --------------------
        # wind_potential_index
        # --------------------
        if "wind_potential_index" in df.columns:
            df["wind_potential_index"] = df["wind_potential_index"]

        # --------------------
        # 直接使用 cloud_cover 和 precipitation 作为特征
        # --------------------
        if "cloud_cover" in df.columns:
            # 确保 cloud_cover 在合理范围内（通常0-100）
            df["cloud_cover"] = df["cloud_cover"].clip(0, 100)
        else:
            print("警告：数据中缺少 'cloud_cover' 列")
            # 设置默认值
            df["cloud_cover"] = 50  # 默认中等云量

        if "precipitation" in df.columns:
            # 确保 precipitation 为非负值
            df["precipitation"] = df["precipitation"].clip(lower=0)
            df["precipitation"] = df["precipitation"] * 10
        else:
            print("警告：数据中缺少 'precipitation' 列")
            # 设置默认值
            df["precipitation"] = 0  # 默认无降水

        # --------------------
        # Season
        # --------------------
        if "season_factor" not in df.columns:
            df["season_factor"] = 1

        # --------------------
        # Feature List
        # --------------------
        keep = [
            "hour", "month", "hour_sin", "hour_cos", "month_sin", "month_cos",
            "hourly__wind_speed_100m", "ws2", "ws3",
            "wind_direction_sin", "wind_direction_cos",
            "ws_lag1", "ws_lag3", "ws_lag6", "ws_roll3", "ws_roll6",
            "wind_potential_index",
            "cloud_cover", "precipitation",  # 直接使用原始天气特征
            "season_factor"
        ]

        return df[[c for c in keep if c in df.columns]]

    # ------------------------------------
    # 训练
    # ------------------------------------
    def fit(self, df):
        features = self._create_features(df)

        self.feature_names = features.columns.tolist()
        target = df["y"]

        # 获取时间权重
        sample_weights = self._get_time_weights(df)

        # 缺失值填补
        features = self.imputer.fit_transform(features)

        self.feature_columns = features.shape[1]
        self.model.fit(features, target, sample_weight=sample_weights)
        self.is_fitted = True
        return self

    # ------------------------------------
    # 预测
    # ------------------------------------
    def predict(self, df):
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit()")

        features = self._create_features(df)
        features = self.imputer.transform(features)

        y_pred = self.model.predict(features)
        y_pred = y_pred

        return pd.DataFrame({
            "ds": df["ds"],
            "yhat": y_pred
        })

    # ------------------------------------
    # 特征重要性
    # ------------------------------------
    def get_feature_importance(self):
        if not self.is_fitted:
            raise ValueError("模型尚未训练！")

        importances = self.model.feature_importances_

        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)