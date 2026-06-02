"""
HU-039 — Modelo LSTM para series temporales climáticas
Predice rendimiento basado en secuencias de 30 días de datos climáticos
Ejecutar: python scripts/train_lstm.py
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = "data"
MODELS_DIR = "data/models"
os.makedirs(MODELS_DIR, exist_ok=True)

CULTIVOS = ["cafe", "cacao"]
DPTOS = ["Antioquia", "Caldas", "Cauca", "Cundinamarca", "Huila",
         "Meta", "Nariño", "Santander", "Tolima", "Valle"]
SUELOS = ["franco", "arcilloso", "arenoso", "franco_arcilloso", "franco_arenoso"]
VARIEDADES = ["mejorada", "tradicional", "hibrida"]
LABRANZA = ["minima", "convencional", "conservacion"]

FEATURES_TEMPORALES = [
    "temp_promedio_c", "temp_maxima_c", "temp_minima_c",
    "precipitacion_mm_90d", "humedad_promedio_pct", "dias_sin_lluvia",
    "velocidad_viento_ms", "radiacion_solar_kwh",
]

FEATURES_ESTATICAS = [
    "cultivo_enc", "departamento_enc", "ph_suelo", "altitud_msnm",
    "materia_organica_pct", "nivel_fertilizacion", "tiene_riego",
    "nivel_control_plagas", "variedad_enc", "area_sembrada_ha",
]

SEQ_LEN = 30


# ── Dataset ──
class CropDataset(Dataset):
    def __init__(self, X_seq, X_static, y):
        self.X_seq = torch.FloatTensor(X_seq)
        self.X_static = torch.FloatTensor(X_static)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_static[idx], self.y[idx]


# ── Modelo LSTM ──
class CropLSTM(nn.Module):
    def __init__(self, n_temporal, n_static, hidden_size=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_temporal,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.static_fc = nn.Sequential(
            nn.Linear(n_static, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq, x_static):
        lstm_out, _ = self.lstm(x_seq)
        # Attention sobre los timesteps
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        static_out = self.static_fc(x_static)
        combined = torch.cat([context, static_out], dim=1)
        return self.fc(combined).squeeze(1)


def build_sequences(df):
    """
    Construye secuencias temporales sintéticas por registro.
    Cada muestra genera 30 timesteps de clima con variación diaria.
    """
    X_seq_list = []
    X_static_list = []
    y_list = []

    le_c = LabelEncoder().fit(CULTIVOS)
    le_d = LabelEncoder().fit(DPTOS)
    le_v = LabelEncoder().fit(VARIEDADES)

    scaler_temporal = MinMaxScaler()
    scaler_static = MinMaxScaler()

    # Fit scalers
    temporal_sample = df[FEATURES_TEMPORALES].values
    scaler_temporal.fit(temporal_sample)
    static_cols = ["cultivo_enc", "departamento_enc", "ph_suelo", "altitud_msnm",
                   "materia_organica_pct", "nivel_fertilizacion", "tiene_riego",
                   "nivel_control_plagas", "variedad_enc", "area_sembrada_ha"]

    df2 = df.copy()
    df2["cultivo_enc"] = le_c.transform(df["cultivo"])
    df2["departamento_enc"] = le_d.transform(df["departamento"])
    df2["variedad_enc"] = le_v.transform(df["variedad"])
    scaler_static.fit(df2[static_cols].values)

    for _, row in df2.iterrows():
        # Generar secuencia de 30 días con variación aleatoria pequeña
        seq = []
        for _ in range(SEQ_LEN):
            timestep = [
                row["temp_promedio_c"] + np.random.normal(0, 0.5),
                row["temp_maxima_c"] + np.random.normal(0, 0.5),
                row["temp_minima_c"] + np.random.normal(0, 0.5),
                max(0, row["precipitacion_mm_90d"] + np.random.normal(0, 2)),
                row["humedad_promedio_pct"] + np.random.normal(0, 1),
                max(0, row["dias_sin_lluvia"] + np.random.randint(-2, 3)),
                max(0, row["velocidad_viento_ms"] + np.random.normal(0, 0.1)),
                max(0, row["radiacion_solar_kwh"] + np.random.normal(0, 0.1)),
            ]
            seq.append(timestep)

        seq_arr = np.array(seq, dtype=np.float32)
        seq_norm = scaler_temporal.transform(seq_arr)
        static_arr = np.array([[
            row["cultivo_enc"], row["departamento_enc"], row["ph_suelo"],
            row["altitud_msnm"], row["materia_organica_pct"], row["nivel_fertilizacion"],
            row["tiene_riego"], row["nivel_control_plagas"],
            row["variedad_enc"], row["area_sembrada_ha"],
        ]], dtype=np.float32)
        static_norm = scaler_static.transform(static_arr)[0]

        X_seq_list.append(seq_norm)
        X_static_list.append(static_norm)
        y_list.append(float(row["rendimiento_ton_ha"]))

    return (np.array(X_seq_list), np.array(X_static_list),
            np.array(y_list), scaler_temporal, scaler_static)


def main():
    print("=== HU-039: Entrenamiento LSTM ===\n")

    df = pd.read_csv(f"{DATA_DIR}/raw/dataset_agrovision_raw.csv")
    print(f"Dataset: {len(df)} registros")

    print("Construyendo secuencias temporales...")
    X_seq, X_static, y, scaler_t, scaler_s = build_sequences(df)
    print(f"Secuencias: {X_seq.shape} | Estáticas: {X_static.shape}")

    X_seq_tr, X_seq_tmp, X_st_tr, X_st_tmp, y_tr, y_tmp = train_test_split(
        X_seq, X_static, y, test_size=0.30, random_state=42)
    X_seq_v, X_seq_te, X_st_v, X_st_te, y_v, y_te = train_test_split(
        X_seq_tmp, X_st_tmp, y_tmp, test_size=0.50, random_state=42)

    print(f"Train: {len(y_tr)} | Val: {len(y_v)} | Test: {len(y_te)}")

    train_ds = CropDataset(X_seq_tr, X_st_tr, y_tr)
    val_ds = CropDataset(X_seq_v, X_st_v, y_v)
    test_ds = CropDataset(X_seq_te, X_st_te, y_te)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)
    test_dl = DataLoader(test_ds, batch_size=64)

    device = torch.device("cpu")
    model = CropLSTM(
        n_temporal=len(FEATURES_TEMPORALES),
        n_static=len(FEATURES_ESTATICAS),
        hidden_size=64,
        n_layers=2,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    print("\nEntrenando LSTM...")
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    PATIENCE = 15
    EPOCHS = 100

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x_seq, x_st, y_batch in train_dl:
            x_seq, x_st, y_batch = x_seq.to(device), x_st.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_seq, x_st)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_seq, x_st, y_batch in val_dl:
                x_seq, x_st, y_batch = x_seq.to(device), x_st.to(device), y_batch.to(device)
                pred = model(x_seq, x_st)
                val_loss += criterion(pred, y_batch).item()

        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping en epoch {epoch+1}")
            break

    # Evaluar en test
    model.load_state_dict(best_state)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_seq, x_st, y_batch in test_dl:
            x_seq, x_st = x_seq.to(device), x_st.to(device)
            pred = model(x_seq, x_st)
            preds.extend(pred.cpu().numpy())
            targets.extend(y_batch.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    mae = float(mean_absolute_error(targets, preds))
    r2 = float(r2_score(targets, preds))
    rmse_pct = rmse / float(np.mean(targets)) * 100

    print(f"\n=== Resultados LSTM ===")
    print(f"  RMSE: {rmse_pct:.2f}% | MAE: {mae:.4f} | R²: {r2:.4f}")

    rmse_ok = bool(rmse_pct < 15)
    r2_ok = bool(r2 > 0.75)
    print(f"\n=== Criterios HU-039 ===")
    print(f"  RMSE < 15%: {'✅' if rmse_ok else '❌'} ({rmse_pct:.2f}%)")
    print(f"  R² > 0.75:  {'✅' if r2_ok else '❌'} ({r2:.4f})")

    # Guardar modelo
    torch.save({
        "model_state": best_state,
        "model_config": {
            "n_temporal": len(FEATURES_TEMPORALES),
            "n_static": len(FEATURES_ESTATICAS),
            "hidden_size": 64,
            "n_layers": 2,
            "dropout": 0.2,
        },
        "features_temporales": FEATURES_TEMPORALES,
        "features_estaticas": FEATURES_ESTATICAS,
        "seq_len": SEQ_LEN,
    }, f"{MODELS_DIR}/lstm_model.pt")

    metrics = {
        "model": "lstm",
        "version": "1.0.0",
        "architecture": "LSTM + Attention + Static features",
        "seq_len": SEQ_LEN,
        "hidden_size": 64,
        "n_layers": 2,
        "test": {
            "rmse": round(rmse, 4),
            "rmse_pct": round(rmse_pct, 2),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
        },
        "criteria_met": {
            "rmse_lt_15pct": rmse_ok,
            "r2_gt_075": r2_ok,
        },
    }

    with open(f"{MODELS_DIR}/lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Modelo: {MODELS_DIR}/lstm_model.pt")
    print(f"✅ Métricas: {MODELS_DIR}/lstm_metrics.json")


if __name__ == "__main__":
    main()