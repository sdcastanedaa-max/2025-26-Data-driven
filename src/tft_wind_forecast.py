"""
TFT Wind Power Forecasting - callable API version

You can now:
    from tft_wind_forecast import prepare_dataframe, build_datasets, train_tft, predict, load_tft_model
    pip install pytorch-lightning pytorch-forecasting torch pandas scikit-learn

And then use:
    model = train_tft(...)
    preds = predict(model, ...)

"""
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE


def prepare_dataframe(df: pd.DataFrame, timestamp_col: str = "ds") -> pd.DataFrame:
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    start = df[timestamp_col].iloc[0]
    df["time_idx"] = ((df[timestamp_col] - start) / pd.Timedelta(hours=1)).astype(int)
    if "series" not in df.columns:
        df["series"] = "site_0"
    df["hour"] = df[timestamp_col].dt.hour
    df["dayofweek"] = df[timestamp_col].dt.dayofweek
    df["dayofyear"] = df[timestamp_col].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"]/24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"]/24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"]/365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"]/365.25)
    return df


def build_datasets(df: pd.DataFrame, target: str = "y", encoder_length: int = 168, decoder_length: int = 24,
                   time_idx: str = "time_idx"):
    exclude = {target, time_idx, "ds", "series"}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

    time_varying_known_reals = ["time_idx", "hour", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    weather_cols = [c for c in numeric_cols if c not in time_varying_known_reals]
    time_varying_known_reals += weather_cols

    training_cutoff = df[time_idx].max() - decoder_length - 7*24

    dataset = TimeSeriesDataSet(
        df[lambda x: x[time_idx] <= training_cutoff],
        time_idx=time_idx,
        target=target,
        group_ids=["series"],
        min_encoder_length=24,
        max_encoder_length=encoder_length,
        min_prediction_length=1,
        max_prediction_length=decoder_length,
        static_categoricals=["series"],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=[target],
        target_normalizer=GroupNormalizer(groups=["series"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    val_cutoff = df[time_idx].max() - decoder_length - 7*24
    training = TimeSeriesDataSet.from_dataset(dataset, df[lambda x: x[time_idx] <= val_cutoff])
    validation = TimeSeriesDataSet.from_dataset(dataset, df[lambda x: x[time_idx] > val_cutoff])

    return dataset, training, validation


def train_tft(training, validation, max_epochs=20, learning_rate=3e-4, checkpoint_path="tft_ckpt"):
    batch_size = 64
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size*2, num_workers=4)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,
        loss=MAE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    ckpt = ModelCheckpoint(dirpath=checkpoint_path, filename="best_tft", save_top_k=1, monitor="val_loss")
    early = EarlyStopping(monitor="val_loss", patience=6, mode="min")

    trainer = Trainer(max_epochs=max_epochs, gpus=1 if torch.cuda.is_available() else 0,
                      gradient_clip_val=0.1, callbacks=[ckpt, early], logger=False)

    trainer.fit(tft, train_loader, val_loader)
    best_path = ckpt.best_model_path
    return TemporalFusionTransformer.load_from_checkpoint(best_path)


def load_tft_model(path: str):
    return TemporalFusionTransformer.load_from_checkpoint(path)


def predict(model, dataset, batch_size=64):
    loader = dataset.to_dataloader(train=False, batch_size=batch_size)
    preds, x = model.predict(loader, return_x=True)
    return preds, x
