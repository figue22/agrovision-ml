"""
HU-037/038 — Feature Engineering Pipeline
Preprocesamiento de features para modelos XGBoost y LSTM
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Any

FEATURES_ESTATICAS = [
    "cultivo_enc", "departamento_enc", "tipo_suelo_enc", "variedad_enc",
    "labranza_enc", "es_permanente",
    "ph_suelo", "altitud_msnm", "materia_organica_pct",
    "nitrogeno_ppm", "fosforo_ppm", "potasio_meq",
    "temp_promedio_c", "temp_maxima_c", "temp_minima_c",
    "precipitacion_mm_90d", "humedad_promedio_pct", "dias_sin_lluvia",
    "velocidad_viento_ms", "radiacion_solar_kwh",
    "area_sembrada_ha", "dias_desde_siembra", "densidad_siembra_rel",
    "nivel_fertilizacion", "tiene_riego", "nivel_control_plagas",
    "amplitud_termica", "indice_humedad", "indice_fertilidad", "score_practicas",
]

FEATURES_TEMPORALES = [
    "temp_promedio_c", "temp_maxima_c", "temp_minima_c",
    "precipitacion_mm_90d", "humedad_promedio_pct", "dias_sin_lluvia",
    "velocidad_viento_ms", "radiacion_solar_kwh",
]

CULTIVOS_VALIDOS = ["cacao", "cafe"]
DEPARTAMENTOS_VALIDOS = [
    "Antioquia", "Caldas", "Cauca", "Cundinamarca", "Huila",
    "Meta", "Nariño", "Santander", "Tolima", "Valle",
]
TIPOS_SUELO = ["franco", "arcilloso", "arenoso", "franco_arcilloso", "franco_arenoso"]
VARIEDADES = ["hibrida", "mejorada", "tradicional"]
LABRANZA = ["conservacion", "convencional", "minima"]


class FeatureEngineering:
    """Pipeline de feature engineering para AgroVision ML."""

    def __init__(self):
        self.le_cultivo = LabelEncoder().fit(CULTIVOS_VALIDOS)
        self.le_dpto = LabelEncoder().fit(DEPARTAMENTOS_VALIDOS)
        self.le_tipo_suelo = LabelEncoder().fit(TIPOS_SUELO)
        self.le_variedad = LabelEncoder().fit(VARIEDADES)
        self.le_labranza = LabelEncoder().fit(LABRANZA)
        self.scaler = StandardScaler()
        self._scaler_fitted = False

    def transform_request(self, data: Dict[str, Any]) -> np.ndarray:
        """Transforma un request de predicción en el vector de features."""
        cultivo = data.get("cultivo", "cafe")
        if cultivo not in CULTIVOS_VALIDOS:
            cultivo = "cafe"

        departamento = data.get("departamento", "Caldas")
        if departamento not in DEPARTAMENTOS_VALIDOS:
            departamento = "Caldas"

        tipo_suelo = data.get("tipo_suelo", "franco")
        if tipo_suelo not in TIPOS_SUELO:
            tipo_suelo = "franco"

        variedad = data.get("variedad", "mejorada")
        if variedad not in VARIEDADES:
            variedad = "mejorada"

        labranza = data.get("tipo_labranza", "convencional")
        if labranza not in LABRANZA:
            labranza = "convencional"

        categoria = data.get("categoria_cultivo", "transitorio")

        cultivo_enc = int(self.le_cultivo.transform([cultivo])[0])
        dpto_enc = int(self.le_dpto.transform([departamento])[0])
        suelo_enc = int(self.le_tipo_suelo.transform([tipo_suelo])[0])
        variedad_enc = int(self.le_variedad.transform([variedad])[0])
        labranza_enc = int(self.le_labranza.transform([labranza])[0])
        es_permanente = 1 if categoria == "permanente" else 0

        ph_suelo = float(data.get("ph_suelo", 6.0))
        altitud = float(data.get("altitud_msnm", 1500))
        materia_org = float(data.get("materia_organica_pct", 3.0))
        nitrogeno = float(data.get("nitrogeno_ppm", 30))
        fosforo = float(data.get("fosforo_ppm", 20))
        potasio = float(data.get("potasio_meq", 0.5))
        temp_prom = float(data.get("temp_promedio_c", 20))
        temp_max = float(data.get("temp_maxima_c", temp_prom + 5))
        temp_min = float(data.get("temp_minima_c", temp_prom - 5))
        precipitacion = float(data.get("precipitacion_mm_90d", 500))
        humedad = float(data.get("humedad_promedio_pct", 75))
        dias_sin_lluvia = int(data.get("dias_sin_lluvia", 5))
        viento = float(data.get("velocidad_viento_ms", 2.0))
        radiacion = float(data.get("radiacion_solar_kwh", 4.5))
        area = float(data.get("area_sembrada_ha", 2.0))
        dias_siembra = int(data.get("dias_desde_siembra", 90))
        densidad = float(data.get("densidad_siembra_rel", 1.0))
        fertilizacion = int(data.get("nivel_fertilizacion", 1))
        riego = int(data.get("tiene_riego", 0))
        control_plagas = int(data.get("nivel_control_plagas", 1))

        amplitud_termica = temp_max - temp_min
        indice_humedad = precipitacion / (dias_sin_lluvia + 1)
        indice_fertilidad = (
            materia_org * 0.4
            + nitrogeno / 60 * 0.3
            + fosforo / 50 * 0.3
        )
        score_practicas = (
            fertilizacion / 2 * 0.4
            + riego * 0.3
            + control_plagas / 2 * 0.3
        )

        features = np.array([[
            cultivo_enc, dpto_enc, suelo_enc, variedad_enc, labranza_enc, es_permanente,
            ph_suelo, altitud, materia_org, nitrogeno, fosforo, potasio,
            temp_prom, temp_max, temp_min, precipitacion, humedad, dias_sin_lluvia,
            viento, radiacion, area, dias_siembra, densidad,
            fertilizacion, riego, control_plagas,
            amplitud_termica, indice_humedad, indice_fertilidad, score_practicas,
        ]], dtype=np.float32)

        return features

    def transform_clima_series(self, datos_clima: list) -> np.ndarray:
        """Transforma serie de datos climáticos para el modelo LSTM."""
        if not datos_clima:
            return np.zeros((1, 30, len(FEATURES_TEMPORALES)), dtype=np.float32)

        rows = []
        for d in datos_clima[-30:]:
            rows.append([
                float(d.get("temp_promedio_c", 20)),
                float(d.get("temp_maxima_c", 25)),
                float(d.get("temp_minima_c", 15)),
                float(d.get("precipitacion_mm_90d", 0)),
                float(d.get("humedad_promedio_pct", 70)),
                float(d.get("dias_sin_lluvia", 0)),
                float(d.get("velocidad_viento_ms", 2)),
                float(d.get("radiacion_solar_kwh", 4)),
            ])

        while len(rows) < 30:
            rows.insert(0, rows[0] if rows else [20, 25, 15, 0, 70, 0, 2, 4])

        arr = np.array(rows, dtype=np.float32)
        return arr.reshape(1, 30, len(FEATURES_TEMPORALES))

    def get_feature_names(self) -> list:
        return FEATURES_ESTATICAS

    def get_temporal_feature_names(self) -> list:
        return FEATURES_TEMPORALES
