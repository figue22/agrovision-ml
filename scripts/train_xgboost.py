"""
HU-038 — Entrenamiento modelos XGBoost separados por cultivo
Un modelo por cultivo para evitar dominancia del feature cultivo_enc
"""

import json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

DATA_DIR = "data"
MODELS_DIR = "data/models"
os.makedirs(MODELS_DIR, exist_ok=True)

CULTIVOS = ["platano", "cacao"]
DPTOS = ["Antioquia","Caldas","Cauca","Cundinamarca","Huila","Meta","Nariño","Santander","Tolima","Valle"]
SUELOS = ["franco","arcilloso","arenoso","franco_arcilloso","franco_arenoso"]
VARIEDADES = ["mejorada","tradicional","hibrida"]
LABRANZA = ["minima","convencional","conservacion"]

# Sin cultivo_enc — cada modelo es específico de un cultivo
FEATURES = [
    "departamento_enc","tipo_suelo_enc","variedad_enc","labranza_enc","es_permanente",
    "ph_suelo","altitud_msnm","materia_organica_pct","nitrogeno_ppm","fosforo_ppm","potasio_meq",
    "temp_promedio_c","temp_maxima_c","temp_minima_c","precipitacion_mm_90d","humedad_promedio_pct",
    "dias_sin_lluvia","velocidad_viento_ms","radiacion_solar_kwh",
    "area_sembrada_ha","dias_desde_siembra","densidad_siembra_rel",
    "nivel_fertilizacion","tiene_riego","nivel_control_plagas",
    "amplitud_termica","indice_humedad","indice_fertilidad","score_practicas",
]


def feature_engineering(df):
    df = df.copy()
    le_d = LabelEncoder().fit(DPTOS)
    le_s = LabelEncoder().fit(SUELOS)
    le_v = LabelEncoder().fit(VARIEDADES)
    le_l = LabelEncoder().fit(LABRANZA)
    df["departamento_enc"] = le_d.transform(df["departamento"])
    df["tipo_suelo_enc"] = le_s.transform(df["tipo_suelo"])
    df["variedad_enc"] = le_v.transform(df["variedad"])
    df["labranza_enc"] = le_l.transform(df["tipo_labranza"])
    df["es_permanente"] = (df["categoria_cultivo"] == "permanente").astype(int)
    df["amplitud_termica"] = df["temp_maxima_c"] - df["temp_minima_c"]
    df["indice_humedad"] = df["precipitacion_mm_90d"] / (df["dias_sin_lluvia"] + 1)
    df["indice_fertilidad"] = (df["materia_organica_pct"]*0.4
        + df["nitrogeno_ppm"]/60*0.3 + df["fosforo_ppm"]/50*0.3)
    df["score_practicas"] = (df["nivel_fertilizacion"]/2*0.4
        + df["tiene_riego"]*0.3 + df["nivel_control_plagas"]/2*0.3)
    return df


def train_model_for_cultivo(df_cultivo, cultivo_name):
    print(f"\n{'='*50}")
    print(f"Entrenando modelo para: {cultivo_name.upper()}")
    print(f"Registros: {len(df_cultivo)}")

    X = df_cultivo[FEATURES]
    y = df_cultivo["rendimiento_ton_ha"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    model = XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.5, random_state=42, n_jobs=-1,
        eval_metric="rmse", early_stopping_rounds=40,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    pred_test = model.predict(X_test)
    r2 = float(r2_score(y_test, pred_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mae = float(mean_absolute_error(y_test, pred_test))
    rmse_pct = rmse / float(y_test.mean()) * 100

    s = "✅" if rmse_pct < 15 and r2 > 0.75 else "❌"
    print(f"\n{s} RMSE: {rmse_pct:.2f}% | MAE: {mae:.4f} | R²: {r2:.4f}")

    # Feature importance
    fi = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 features:")
    for feat, imp in fi:
        print(f"  {feat:35s}: {'█' * int(imp * 40)} {imp:.4f}")

    # Guardar modelo
    model_path = f"{MODELS_DIR}/xgboost_{cultivo_name}.joblib"
    joblib.dump(model, model_path)
    print(f"\n✅ Modelo guardado: {model_path}")

    return {
        "rmse": round(rmse, 4),
        "rmse_pct": round(rmse_pct, 2),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "n_test": int(len(X_test)),
        "n_estimators": int(model.best_iteration),
        "feature_importance": {k: round(float(v), 4) for k, v in fi},
    }


def main():
    print("=== HU-038: Entrenamiento XGBoost por cultivo ===\n")

    df = feature_engineering(pd.read_csv(f"{DATA_DIR}/raw/dataset_agrovision_raw.csv"))
    print(f"Dataset total: {len(df)} registros")
    print(f"Cultivos: {df['cultivo'].value_counts().to_dict()}\n")

    metrics_por_cultivo = {}
    all_ok = True

    for cultivo in CULTIVOS:
        df_c = df[df["cultivo"] == cultivo].reset_index(drop=True)
        if len(df_c) < 100:
            print(f"⚠️  {cultivo}: muy pocos registros ({len(df_c)}), saltando")
            continue

        m = train_model_for_cultivo(df_c, cultivo)
        metrics_por_cultivo[cultivo] = m
        if m["rmse_pct"] >= 15 or m["r2"] < 0.75:
            all_ok = False

    # Guardar métricas globales
    metrics = {
        "model": "xgboost_por_cultivo",
        "version": "2.0.0",
        "estrategia": "modelo_separado_por_cultivo",
        "cultivos": CULTIVOS,
        "features": FEATURES,
        "metricas_por_cultivo": metrics_por_cultivo,
        "criteria_met": {
            "todos_rmse_lt_15pct": all_ok,
            "todos_r2_gt_075": all_ok,
        },
        "nota": "Modelos separados por cultivo eliminan dominancia de cultivo_enc",
    }

    with open(f"{MODELS_DIR}/xgboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print("RESUMEN FINAL:")
    for cn, m in metrics_por_cultivo.items():
        s = "✅" if m["rmse_pct"] < 15 and m["r2"] > 0.75 else "❌"
        print(f"  {s} {cn}: RMSE {m['rmse_pct']:.1f}% | R² {m['r2']:.3f}")
    print(f"\n✅ Métricas: {MODELS_DIR}/xgboost_metrics.json")


if __name__ == "__main__":
    main()