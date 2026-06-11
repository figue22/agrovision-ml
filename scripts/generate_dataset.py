"""
HU-037 — Generador de Dataset Cultivos Colombianos
Basado en rangos reales de AGRONET, FAO y AGROSAVIA Colombia
Ejecutar: python scripts/generate_dataset.py
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

np.random.seed(42)
N_SAMPLES = 8000

CULTIVOS = {
    "platano": {
        "rend": (10.0, 40.0),
        "temp": (22, 30),
        "alt": (0, 1500),
        "ph": (5.5, 7.0),
        "agua": (1200, 2500),
        "cat": "permanente",
        "w": 0.55,
        "dias_crecimiento": (270, 365),
    },
    "cacao": {
        "rend": (0.2, 1.8),
        "temp": (24, 30),
        "alt": (0, 1200),
        "ph": (5.0, 7.0),
        "agua": (1500, 2500),
        "cat": "permanente",
        "w": 0.45,
        "dias_crecimiento": (150, 180),
    },
}

DEPARTAMENTOS = {
    "Caldas":       {"altitud_media": 1500, "precipitacion_media": 2200, "temp_media": 20},
    "Antioquia":    {"altitud_media": 1500, "precipitacion_media": 2400, "temp_media": 21},
    "Cundinamarca": {"altitud_media": 2000, "precipitacion_media": 1000, "temp_media": 14},
    "Nariño":       {"altitud_media": 2500, "precipitacion_media": 1500, "temp_media": 13},
    "Valle":        {"altitud_media": 1000, "precipitacion_media": 1800, "temp_media": 24},
    "Tolima":       {"altitud_media": 1000, "precipitacion_media": 1400, "temp_media": 22},
    "Huila":        {"altitud_media": 900,  "precipitacion_media": 1500, "temp_media": 23},
    "Cauca":        {"altitud_media": 1800, "precipitacion_media": 2000, "temp_media": 18},
    "Santander":    {"altitud_media": 1200, "precipitacion_media": 1800, "temp_media": 22},
    "Meta":         {"altitud_media": 300,  "precipitacion_media": 2800, "temp_media": 27},
}

def generate_record(cultivo_name, c, dpto_name, d):
    ph_suelo = np.random.uniform(4.0, 8.0)
    altitud = max(0, d["altitud_media"] + np.random.normal(0, 300))
    tipo_suelo = np.random.choice(["franco", "arcilloso", "arenoso", "franco_arcilloso", "franco_arenoso"], p=[0.35, 0.25, 0.15, 0.15, 0.10])
    materia_org = np.random.uniform(1.0, 6.0)
    nitrogeno = np.random.uniform(10, 60)
    fosforo = np.random.uniform(5, 50)
    potasio = np.random.uniform(0.1, 1.5)
    temp_prom = d["temp_media"] + np.random.normal(0, 3)
    temp_max = temp_prom + np.random.uniform(3, 8)
    temp_min = temp_prom - np.random.uniform(3, 8)
    precipitacion = d["precipitacion_media"] * np.random.uniform(0.5, 1.5) * (90/365)
    humedad = np.random.uniform(55, 90)
    dias_sin_lluvia = np.random.randint(0, 30)
    viento = np.random.uniform(0.5, 5.0)
    radiacion = np.random.uniform(3.0, 6.5)
    area = np.random.uniform(0.5, 20.0)
    dias_siembra = np.random.randint(30, 300)
    variedad = np.random.choice(["mejorada", "tradicional", "hibrida"], p=[0.4, 0.35, 0.25])
    densidad = np.random.uniform(0.5, 2.0)
    fertilizacion = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
    riego = np.random.choice([0, 1], p=[0.4, 0.6])
    control_plagas = np.random.choice([0, 1, 2], p=[0.15, 0.55, 0.30])
    labranza = np.random.choice(["minima", "convencional", "conservacion"], p=[0.3, 0.45, 0.25])

    # ── Rendimiento determinístico con scores continuos ──
    rend_min, rend_max = c["rend"]

    ph_min, ph_max = c["ph"]
    score_ph = 1.0 if ph_min <= ph_suelo <= ph_max else max(0.1, 1 - abs(ph_suelo - (ph_min + ph_max) / 2) / 3)

    t_min_c, t_max_c = c["temp"]
    score_temp = 1.0 if t_min_c <= temp_prom <= t_max_c else max(0.1, 1 - abs(temp_prom - (t_min_c + t_max_c) / 2) / 10)

    agua_min_90d = c["agua"][0] * 90 / 365
    agua_max_90d = c["agua"][1] * 90 / 365
    prec_medio = (agua_min_90d + agua_max_90d) / 2
    score_agua = max(0.1, 1 - abs(precipitacion - prec_medio) / (prec_medio + 1))

    alt_min, alt_max = c["alt"]
    alt_medio = (alt_min + alt_max) / 2
    score_alt = max(0.1, 1 - abs(altitud - alt_medio) / (alt_medio + 1))

    score_practicas_val = fertilizacion / 2 * 0.4 + riego * 0.3 + control_plagas / 2 * 0.3
    score_variedad = {"mejorada": 0.8, "tradicional": 0.5, "hibrida": 1.0}[variedad]

    score_total = (score_ph * 0.20 + score_temp * 0.25 + score_agua * 0.20 +
                   score_alt * 0.15 + score_practicas_val * 0.12 + score_variedad * 0.08)

    rendimiento = rend_min + score_total * (rend_max - rend_min)
    # Ruido mínimo: 0.5%
    rendimiento = max(rend_min * 0.5, rendimiento + np.random.normal(0, rendimiento * 0.005))

    return {
        "cultivo": cultivo_name, "departamento": dpto_name, "categoria_cultivo": c["cat"],
        "ph_suelo": round(ph_suelo, 2), "altitud_msnm": round(altitud, 0), "tipo_suelo": tipo_suelo,
        "materia_organica_pct": round(materia_org, 2), "nitrogeno_ppm": round(nitrogeno, 1),
        "fosforo_ppm": round(fosforo, 1), "potasio_meq": round(potasio, 3),
        "temp_promedio_c": round(temp_prom, 1), "temp_maxima_c": round(temp_max, 1), "temp_minima_c": round(temp_min, 1),
        "precipitacion_mm_90d": round(precipitacion, 1), "humedad_promedio_pct": round(humedad, 1),
        "dias_sin_lluvia": dias_sin_lluvia, "velocidad_viento_ms": round(viento, 2), "radiacion_solar_kwh": round(radiacion, 2),
        "area_sembrada_ha": round(area, 2), "dias_desde_siembra": dias_siembra, "variedad": variedad,
        "densidad_siembra_rel": round(densidad, 2), "nivel_fertilizacion": fertilizacion, "tiene_riego": riego,
        "nivel_control_plagas": control_plagas, "tipo_labranza": labranza,
        "rendimiento_ton_ha": round(rendimiento, 3),
    }


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)

    cultivos_list = list(CULTIVOS.keys())
    pesos = [CULTIVOS[c]["w"] for c in cultivos_list]

    records = []
    for _ in range(N_SAMPLES):
        cultivo = np.random.choice(cultivos_list, p=pesos)
        dpto = np.random.choice(list(DEPARTAMENTOS.keys()))
        records.append(generate_record(cultivo, CULTIVOS[cultivo], dpto, DEPARTAMENTOS[dpto]))

    df = pd.DataFrame(records)
    df.to_csv("data/raw/dataset_agrovision_raw.csv", index=False)
    print(f"✅ Dataset raw: {len(df)} registros → data/raw/dataset_agrovision_raw.csv")

    # Feature engineering
    le_cultivo = LabelEncoder().fit(cultivos_list)
    le_dpto = LabelEncoder().fit(list(DEPARTAMENTOS.keys()))
    le_suelo = LabelEncoder().fit(["franco", "arcilloso", "arenoso", "franco_arcilloso", "franco_arenoso"])
    le_variedad = LabelEncoder().fit(["mejorada", "tradicional", "hibrida"])
    le_labranza = LabelEncoder().fit(["minima", "convencional", "conservacion"])

    df["cultivo_enc"] = le_cultivo.transform(df["cultivo"])
    df["departamento_enc"] = le_dpto.transform(df["departamento"])
    df["tipo_suelo_enc"] = le_suelo.transform(df["tipo_suelo"])
    df["variedad_enc"] = le_variedad.transform(df["variedad"])
    df["labranza_enc"] = le_labranza.transform(df["tipo_labranza"])
    df["es_permanente"] = (df["categoria_cultivo"] == "permanente").astype(int)
    df["amplitud_termica"] = df["temp_maxima_c"] - df["temp_minima_c"]
    df["indice_humedad"] = df["precipitacion_mm_90d"] / (df["dias_sin_lluvia"] + 1)
    df["indice_fertilidad"] = df["materia_organica_pct"] * 0.4 + df["nitrogeno_ppm"] / 60 * 0.3 + df["fosforo_ppm"] / 50 * 0.3
    df["score_practicas"] = df["nivel_fertilizacion"] / 2 * 0.4 + df["tiene_riego"] * 0.3 + df["nivel_control_plagas"] / 2 * 0.3

    FEATURES = [
        "cultivo_enc", "departamento_enc", "tipo_suelo_enc", "variedad_enc", "labranza_enc", "es_permanente",
        "ph_suelo", "altitud_msnm", "materia_organica_pct", "nitrogeno_ppm", "fosforo_ppm", "potasio_meq",
        "temp_promedio_c", "temp_maxima_c", "temp_minima_c", "precipitacion_mm_90d", "humedad_promedio_pct",
        "dias_sin_lluvia", "velocidad_viento_ms", "radiacion_solar_kwh",
        "area_sembrada_ha", "dias_desde_siembra", "densidad_siembra_rel",
        "nivel_fertilizacion", "tiene_riego", "nivel_control_plagas",
        "amplitud_termica", "indice_humedad", "indice_fertilidad", "score_practicas",
    ]

    X = df[FEATURES]
    y = df["rendimiento_ton_ha"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    df[FEATURES + ["rendimiento_ton_ha"]].to_csv("data/processed/dataset_train.csv", index=False)
    X_train.to_csv("data/splits/X_train.csv", index=False)
    X_val.to_csv("data/splits/X_val.csv", index=False)
    X_test.to_csv("data/splits/X_test.csv", index=False)
    y_train.to_csv("data/splits/y_train.csv", index=False)
    y_val.to_csv("data/splits/y_val.csv", index=False)
    y_test.to_csv("data/splits/y_test.csv", index=False)

    info = {
        "n_samples": len(df), "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "features": FEATURES, "target": "rendimiento_ton_ha",
        "cultivos": cultivos_list,
        "departamentos": list(DEPARTAMENTOS.keys()),
        "split": "70/15/15",
        "fuentes": ["AGRONET Colombia", "FAO FAOSTAT", "AGROSAVIA", "datos sintéticos basados en rangos reales"],
    }
    with open("data/processed/dataset_info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"✅ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"✅ Features: {len(FEATURES)}")
    print("✅ Dataset listo para entrenamiento")


if __name__ == "__main__":
    main()