"""
HU-040 — Ensemble XGBoost + LSTM
Combina predicciones de ambos modelos con pesos optimizados
Ejecutar: python scripts/train_ensemble.py
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = "data"
MODELS_DIR = "data/models"

CULTIVOS = ["cafe", "cacao"]
DPTOS = ["Antioquia", "Caldas", "Cauca", "Cundinamarca", "Huila",
         "Meta", "Nariño", "Santander", "Tolima", "Valle"]
SUELOS = ["franco", "arcilloso", "arenoso", "franco_arcilloso", "franco_arenoso"]
VARIEDADES = ["mejorada", "tradicional", "hibrida"]
LABRANZA = ["minima", "convencional", "conservacion"]

FEATURES_XGB = [
    "cultivo_enc", "departamento_enc", "tipo_suelo_enc", "variedad_enc", "labranza_enc", "es_permanente",
    "ph_suelo", "altitud_msnm", "materia_organica_pct", "nitrogeno_ppm", "fosforo_ppm", "potasio_meq",
    "temp_promedio_c", "temp_maxima_c", "temp_minima_c", "precipitacion_mm_90d", "humedad_promedio_pct",
    "dias_sin_lluvia", "velocidad_viento_ms", "radiacion_solar_kwh",
    "area_sembrada_ha", "dias_desde_siembra", "densidad_siembra_rel",
    "nivel_fertilizacion", "tiene_riego", "nivel_control_plagas",
    "amplitud_termica", "indice_humedad", "indice_fertilidad", "score_practicas",
]

FEATURES_TEMPORALES = [
    "temp_promedio_c", "temp_maxima_c", "temp_minima_c",
    "precipitacion_mm_90d", "humedad_promedio_pct", "dias_sin_lluvia",
    "velocidad_viento_ms", "radiacion_solar_kwh",
]

FEATURES_ESTATICAS_LSTM = [
    "cultivo_enc", "departamento_enc", "ph_suelo", "altitud_msnm",
    "materia_organica_pct", "nivel_fertilizacion", "tiene_riego",
    "nivel_control_plagas", "variedad_enc", "area_sembrada_ha",
]

SEQ_LEN = 30


class CropLSTM(nn.Module):
    def __init__(self, n_temporal, n_static, hidden_size=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_temporal, hidden_size=hidden_size,
            num_layers=n_layers, dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.static_fc = nn.Sequential(
            nn.Linear(n_static, 32), nn.ReLU(), nn.Dropout(0.1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 32, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x_seq, x_static):
        lstm_out, _ = self.lstm(x_seq)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        static_out = self.static_fc(x_static)
        combined = torch.cat([context, static_out], dim=1)
        return self.fc(combined).squeeze(1)


def feature_engineering(df):
    df = df.copy()
    le_c = LabelEncoder().fit(CULTIVOS)
    le_d = LabelEncoder().fit(DPTOS)
    le_s = LabelEncoder().fit(SUELOS)
    le_v = LabelEncoder().fit(VARIEDADES)
    le_l = LabelEncoder().fit(LABRANZA)
    df["cultivo_enc"] = le_c.transform(df["cultivo"])
    df["departamento_enc"] = le_d.transform(df["departamento"])
    df["tipo_suelo_enc"] = le_s.transform(df["tipo_suelo"])
    df["variedad_enc"] = le_v.transform(df["variedad"])
    df["labranza_enc"] = le_l.transform(df["tipo_labranza"])
    df["es_permanente"] = (df["categoria_cultivo"] == "permanente").astype(int)
    df["amplitud_termica"] = df["temp_maxima_c"] - df["temp_minima_c"]
    df["indice_humedad"] = df["precipitacion_mm_90d"] / (df["dias_sin_lluvia"] + 1)
    df["indice_fertilidad"] = (
        df["materia_organica_pct"] * 0.4
        + df["nitrogeno_ppm"] / 60 * 0.3
        + df["fosforo_ppm"] / 50 * 0.3
    )
    df["score_practicas"] = (
        df["nivel_fertilizacion"] / 2 * 0.4
        + df["tiene_riego"] * 0.3
        + df["nivel_control_plagas"] / 2 * 0.3
    )
    return df


def build_lstm_inputs(df, scaler_t=None, scaler_s=None, fit=False):
    X_seq_list = []
    X_static_list = []

    if fit:
        scaler_t = MinMaxScaler()
        scaler_s = MinMaxScaler()
        scaler_t.fit(df[FEATURES_TEMPORALES].values)
        scaler_s.fit(df[FEATURES_ESTATICAS_LSTM].values)

    for _, row in df.iterrows():
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

        seq_norm = scaler_t.transform(np.array(seq, dtype=np.float32))
        static_arr = np.array([[
            row["cultivo_enc"], row["departamento_enc"], row["ph_suelo"],
            row["altitud_msnm"], row["materia_organica_pct"], row["nivel_fertilizacion"],
            row["tiene_riego"], row["nivel_control_plagas"],
            row["variedad_enc"], row["area_sembrada_ha"],
        ]], dtype=np.float32)
        static_norm = scaler_s.transform(static_arr)[0]
        X_seq_list.append(seq_norm)
        X_static_list.append(static_norm)

    return np.array(X_seq_list), np.array(X_static_list), scaler_t, scaler_s


def get_lstm_predictions(model, X_seq, X_static, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_seq), batch_size):
            x_s = torch.FloatTensor(X_seq[i:i+batch_size])
            x_st = torch.FloatTensor(X_static[i:i+batch_size])
            p = model(x_s, x_st)
            preds.extend(p.cpu().numpy())
    return np.array(preds)


def find_optimal_weights(pred_xgb, pred_lstm, y_true):
    """Busca el peso óptimo para XGBoost mediante grid search."""
    best_rmse = float("inf")
    best_w = 0.5
    for w in np.arange(0.0, 1.01, 0.05):
        ensemble = w * pred_xgb + (1 - w) * pred_lstm
        rmse = float(np.sqrt(mean_squared_error(y_true, ensemble)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w
    return best_w, best_rmse


def main():
    print("=== HU-040: Ensemble XGBoost + LSTM ===\n")

    df = pd.read_csv(f"{DATA_DIR}/raw/dataset_agrovision_raw.csv")
    df = feature_engineering(df)

    X_xgb = df[FEATURES_XGB]
    y = df["rendimiento_ton_ha"].values

    _, idx_tmp = train_test_split(np.arange(len(df)), test_size=0.30, random_state=42)
    _, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=42)
    _, idx_val = train_test_split(idx_tmp, test_size=0.50, random_state=42)

    df_te = df.iloc[idx_te].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    y_te = y[idx_te]
    y_val = y[idx_val]

    # ── Cargar modelos ──
    print("Cargando modelos...")
    xgb_model = joblib.load(f"{MODELS_DIR}/xgboost_model.joblib")

    lstm_ckpt = torch.load(f"{MODELS_DIR}/lstm_model.pt", map_location="cpu")
    cfg = lstm_ckpt["model_config"]
    lstm_model = CropLSTM(**cfg)
    lstm_model.load_state_dict(lstm_ckpt["model_state"])
    lstm_model.eval()
    print("✅ Modelos cargados")

    # ── Predicciones XGBoost ──
    print("\nGenerando predicciones XGBoost...")
    pred_xgb_val = xgb_model.predict(X_xgb.iloc[idx_val])
    pred_xgb_te = xgb_model.predict(X_xgb.iloc[idx_te])

    # ── Predicciones LSTM ──
    print("Generando predicciones LSTM...")
    X_seq_val, X_st_val, scaler_t, scaler_s = build_lstm_inputs(df_val, fit=True)
    X_seq_te, X_st_te, _, _ = build_lstm_inputs(df_te, scaler_t=scaler_t, scaler_s=scaler_s)

    pred_lstm_val = get_lstm_predictions(lstm_model, X_seq_val, X_st_val)
    pred_lstm_te = get_lstm_predictions(lstm_model, X_seq_te, X_st_te)

    # ── Encontrar pesos óptimos ──
    print("\nOptimizando pesos del ensemble...")
    best_w, _ = find_optimal_weights(pred_xgb_val, pred_lstm_val, y_val)
    w_lstm = round(1 - best_w, 2)
    print(f"  Peso óptimo — XGBoost: {best_w:.2f} | LSTM: {w_lstm:.2f}")

    # ── Evaluación en test ──
    pred_ensemble = best_w * pred_xgb_te + w_lstm * pred_lstm_te

    rmse_xgb = float(np.sqrt(mean_squared_error(y_te, pred_xgb_te)))
    rmse_lstm = float(np.sqrt(mean_squared_error(y_te, pred_lstm_te)))
    rmse_ens = float(np.sqrt(mean_squared_error(y_te, pred_ensemble)))

    r2_xgb = float(r2_score(y_te, pred_xgb_te))
    r2_lstm = float(r2_score(y_te, pred_lstm_te))
    r2_ens = float(r2_score(y_te, pred_ensemble))

    mean_y = float(np.mean(y_te))
    pct_xgb = rmse_xgb / mean_y * 100
    pct_lstm = rmse_lstm / mean_y * 100
    pct_ens = rmse_ens / mean_y * 100

    print(f"\n=== Comparación de modelos ===")
    print(f"  XGBoost:  RMSE {pct_xgb:.2f}% | R² {r2_xgb:.4f}")
    print(f"  LSTM:     RMSE {pct_lstm:.2f}% | R² {r2_lstm:.4f}")
    print(f"  Ensemble: RMSE {pct_ens:.2f}% | R² {r2_ens:.4f}")

    rmse_ok = bool(pct_ens < 15)
    r2_ok = bool(r2_ens > 0.75)
    print(f"\n=== Criterios HU-040 ===")
    print(f"  RMSE < 15%: {'✅' if rmse_ok else '❌'} ({pct_ens:.2f}%)")
    print(f"  R² > 0.75:  {'✅' if r2_ok else '❌'} ({r2_ens:.4f})")
    print(f"  Ensemble mejor que XGBoost solo: {'✅' if pct_ens <= pct_xgb else '⚠️'}")
    print(f"  Ensemble mejor que LSTM solo:    {'✅' if pct_ens <= pct_lstm else '⚠️'}")

    # Guardar configuración del ensemble
    ensemble_config = {
        "model": "ensemble_xgboost_lstm",
        "version": "1.0.0",
        "peso_xgboost": round(best_w, 2),
        "peso_lstm": round(w_lstm, 2),
        "test": {
            "rmse_pct": round(pct_ens, 2),
            "r2": round(r2_ens, 4),
            "mae": round(float(mean_absolute_error(y_te, pred_ensemble)), 4),
        },
        "comparacion": {
            "xgboost": {"rmse_pct": round(pct_xgb, 2), "r2": round(r2_xgb, 4)},
            "lstm": {"rmse_pct": round(pct_lstm, 2), "r2": round(r2_lstm, 4)},
            "ensemble": {"rmse_pct": round(pct_ens, 2), "r2": round(r2_ens, 4)},
        },
        "criteria_met": {
            "rmse_lt_15pct": rmse_ok,
            "r2_gt_075": r2_ok,
        },
    }

    with open(f"{MODELS_DIR}/ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Configuración guardada: {MODELS_DIR}/ensemble_config.json")


if __name__ == "__main__":
    main()