import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class LightGBM:
    """
    修复版：兼容 train_df 的 LightGBM 风电预测模型（增强对字符/布尔列的容错）
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

    # ------------------------------------------------------------------
    # 修复的特征工程（更鲁棒地处理字符串和布尔列）
    # ------------------------------------------------------------------
    def _create_advanced_features(self, df):
        df = df.copy()

        # 关键修复：训练集有 y，但预测集没有；避免 y 被当作特征
        if 'y' in df.columns:
            df = df.drop(columns=['y'], errors='ignore')

        # 1. 统一 datetime
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
        else:
            raise ValueError("Must have 'ds' column(datetime)")

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

        # 8. 直接使用 cloud_cover 和 precipitation 作为天气特征（替代旧的分类方法）
        # cloud_cover: 云量百分比 (0-100)
        if 'cloud_cover' in df.columns:
            df['cloud_cover'] = pd.to_numeric(df['cloud_cover'], errors='coerce').clip(0, 100)
            # 创建云量的滞后特征
            df['cloud_cover_lag_1'] = df['cloud_cover'].shift(1)
            df['cloud_cover_roll_mean_3'] = df['cloud_cover'].rolling(3, min_periods=1).mean()
        else:
            print("warning: no 'cloud_cover' detected")
            # 设置默认值
            df['cloud_cover'] = 50  # 默认中等云量

        # precipitation: 降水量 (mm/h)
        if 'precipitation' in df.columns:
            df['precipitation'] = pd.to_numeric(df['precipitation'], errors='coerce').clip(lower=0)
            # 创建降水量的滞后特征
            df['precipitation_lag_1'] = df['precipitation'].shift(1)
            df['precipitation_roll_mean_6'] = df['precipitation'].rolling(6, min_periods=1).mean()
            # 降水是否存在的标志
            df['has_precipitation'] = (df['precipitation'] > 0.1).astype(int)
        else:
            print("warning: no 'precipitation' detected")
            # 设置默认值
            df['precipitation'] = 0  # 默认无降水
            df['has_precipitation'] = 0

        # 9. 创建云量和降水的交互特征
        df['cloud_precip_interaction'] = df['cloud_cover'] * (df['precipitation'] + 1)  # 加1避免0值

        # 10. 其他文本/布尔 → 数值
        for col in df.columns:
            if col in ['daily__weather_code', 'hourly__time', 'ds']:
                continue
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)
            elif df[col].dtype == 'object':
                df[col] = df[col].astype('category').cat.codes

        # 11. 删除无用列（包括旧的weather分类特征）
        drop_columns = ['ds', 'hourly__time', 'weekday', 'season', 'daily__weather_code']
        df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

        # 只保留数值
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        return numeric_df

    # ------------------------------------------------------------------
    # 样本权重（仍从原始 df 的 ds 列计算）
    # ------------------------------------------------------------------
    def _get_sample_weights(self, df):
        hours = pd.to_datetime(df['ds']).dt.hour
        weights = np.ones(len(df))
        weights[(hours >= 10) & (hours <= 20)] = 1.0
        weights[(hours >= 14) & (hours <= 16)] = 1.0
        return weights

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def fit(self, df, eval_df=None, early_stopping_rounds=50):
        print('Get Feature...')
        features = self._create_advanced_features(df)
        self.feature_names = features.columns.tolist()
        target = pd.to_numeric(df['y'], errors='coerce').values

        sample_weights = self._get_sample_weights(df)

        print('Deal Loss...')
        features = self.imputer.fit_transform(features)

        print('Standerize Features ...')
        features = self.scaler.fit_transform(features)

        lgb_train = lgb.Dataset(
            features, target,
            weight=sample_weights,
            feature_name=self.feature_names,
            free_raw_data=False
        )

        eval_sets = [lgb_train]
        eval_names = ['train']

        if eval_df is not None:
            eval_features = self._create_advanced_features(eval_df)
            eval_features = self.imputer.transform(eval_features)
            eval_features = self.scaler.transform(eval_features)
            eval_target = pd.to_numeric(eval_df['y'], errors='coerce').values

            lgb_eval = lgb.Dataset(
                eval_features, eval_target,
                reference=lgb_train,
                feature_name=self.feature_names,
                free_raw_data=False
            )
            eval_sets.append(lgb_eval)
            eval_names.append('valid')

        print('Starting LightGBM...')
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
        print('Training Finish')
        return self

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def predict(self, df):
        if not self.is_fitted:
            raise ValueError('Model NOT Trained')

        features = self._create_advanced_features(df)
        features = self.imputer.transform(features)
        features = self.scaler.transform(features)

        pred = self.model.predict(features)

        return pd.DataFrame({
            'ds': df['ds'],
            'yhat': pred
        })

    # ------------------------------------------------------------------
    # 特征重要性
    # ------------------------------------------------------------------
    def get_feature_importance(self):
        if self.model is None:
            raise ValueError('Model NOT Trained')

        imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)
        return imp