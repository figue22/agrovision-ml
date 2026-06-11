"""
HU-041 — Servicio de predicción con modelos reales XGBoost + LSTM + Ensemble
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    NivelRiesgo,
    TipoModelo,
)

logger = logging.getLogger("agrovision-ml")

CULTIVOS = ["platano", "cacao"]
DPTOS = ["Antioquia", "Caldas", "Cauca", "Cundinamarca", "Huila",
         "Meta", "Nariño", "Santander", "Tolima", "Valle"]
SUELOS = ["franco", "arcilloso", "arenoso", "franco_arcilloso", "franco_arenoso"]
VARIEDADES = ["mejorada", "tradicional", "hibrida"]
LABRANZA = ["minima", "convencional", "conservacion"]

FEATURES_XGB = [
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

FEATURES_ESTATICAS_LSTM = [
    "cultivo_enc", "departamento_enc", "ph_suelo", "altitud_msnm",
    "materia_organica_pct", "nivel_fertilizacion", "tiene_riego",
    "nivel_control_plagas", "variedad_enc", "area_sembrada_ha",
]

SEQ_LEN = 30

# ── Parámetros agronómicos reales por cultivo (AGRONET / AGROSAVIA Colombia) ──
CULTIVO_PARAMS: dict = {
    "platano": {
        "ciclo_base": 300, "ciclo_min": 240, "ciclo_max": 420,
        "temp_ref": 26.0, "gdd_base": 14.0,
        "ajuste_alt_cada_100m": 4, "alt_ref": 500,
        "rend_optimo": 30.0, "rend_min_absoluto": 6.0,
    },
    "cacao": {
        "ciclo_base": 160, "ciclo_min": 140, "ciclo_max": 200,
        "temp_ref": 27.0, "gdd_base": 15.0,
        "ajuste_alt_cada_100m": 3, "alt_ref": 300,
        "rend_optimo": 1.2, "rend_min_absoluto": 0.15,
    },
}



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


class PredictionService:
    """Servicio de predicción con modelos reales XGBoost + LSTM + Ensemble."""

    MODEL_VERSION = "1.0.0"

    def __init__(self):
        self._start_time = time.time()
        self._xgb_model = None
        self._lstm_model = None
        self._lstm_config = None
        self._ensemble_config = None
        self._xgb_metrics = {}
        self._lstm_metrics = {}
        self._scaler_t = MinMaxScaler()
        self._scaler_s = MinMaxScaler()
        self._scalers_fitted = False
        self._models_loaded = {"xgboost": False, "lstm": False, "ensemble": False}
        self._load_models()

    def _load_models(self) -> None:
        model_path = os.environ.get("MODEL_PATH", "data/models")

        # XGBoost — modelos separados por cultivo
        self._xgb_models = {}
        for cultivo in CULTIVOS:
            xgb_path = os.path.join(model_path, f"xgboost_{cultivo}.joblib")
            if os.path.exists(xgb_path):
                try:
                    self._xgb_models[cultivo] = joblib.load(xgb_path)
                    logger.info("✅ XGBoost[%s] cargado: %s", cultivo, xgb_path)
                except Exception as e:
                    logger.error("Error cargando XGBoost[%s]: %s", cultivo, e)

        if self._xgb_models:
            self._models_loaded["xgboost"] = True
            metrics_path = os.path.join(model_path, "xgboost_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    self._xgb_metrics = json.load(f)

        # LSTM
        lstm_path = os.path.join(model_path, "lstm_model.pt")
        if os.path.exists(lstm_path):
            try:
                ckpt = torch.load(lstm_path, map_location="cpu")
                self._lstm_config = ckpt["model_config"]
                self._lstm_model = CropLSTM(**self._lstm_config)
                self._lstm_model.load_state_dict(ckpt["model_state"])
                self._lstm_model.eval()
                self._models_loaded["lstm"] = True
                logger.info("✅ LSTM cargado: %s", lstm_path)

                metrics_path = os.path.join(model_path, "lstm_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        self._lstm_metrics = json.load(f)
            except Exception as e:
                logger.error("Error cargando LSTM: %s", e)

        # Ensemble config
        ens_path = os.path.join(model_path, "ensemble_config.json")
        if os.path.exists(ens_path):
            try:
                with open(ens_path) as f:
                    self._ensemble_config = json.load(f)
                if self._models_loaded["xgboost"] or self._models_loaded["lstm"]:
                    self._models_loaded["ensemble"] = True
                logger.info("✅ Ensemble config cargado")
            except Exception as e:
                logger.error("Error cargando ensemble config: %s", e)

    def _encode_features(self, data: dict) -> np.ndarray:
        """Transforma el request en vector de features para XGBoost."""
        cultivo = data.get("cultivo", "platano")
        if cultivo not in CULTIVOS:
            cultivo = "platano"
        departamento = data.get("departamento", "Caldas")
        if departamento not in DPTOS:
            departamento = "Caldas"
        tipo_suelo = data.get("tipo_suelo", "franco")
        if tipo_suelo not in SUELOS:
            tipo_suelo = "franco"
        variedad = data.get("variedad", "mejorada")
        if variedad not in VARIEDADES:
            variedad = "mejorada"
        labranza = data.get("tipo_labranza", "convencional")
        if labranza not in LABRANZA:
            labranza = "convencional"

        cultivo_enc = LabelEncoder().fit(CULTIVOS).transform([cultivo])[0]
        dpto_enc = LabelEncoder().fit(DPTOS).transform([departamento])[0]
        suelo_enc = LabelEncoder().fit(SUELOS).transform([tipo_suelo])[0]
        variedad_enc = LabelEncoder().fit(VARIEDADES).transform([variedad])[0]
        labranza_enc = LabelEncoder().fit(LABRANZA).transform([labranza])[0]
        es_permanente = 1

        ph = float(data.get("ph_suelo", 6.0))
        alt = float(data.get("altitud_msnm", 1500))
        mat_org = float(data.get("materia_organica_pct", 3.0))
        nit = float(data.get("nitrogeno_ppm", 30))
        fos = float(data.get("fosforo_ppm", 20))
        pot = float(data.get("potasio_meq", 0.5))
        temp = float(data.get("temp_promedio_c", 20))
        t_max = float(data.get("temp_maxima_c", temp + 5))
        t_min = float(data.get("temp_minima_c", temp - 5))
        prec = float(data.get("precipitacion_mm_90d", 500))
        hum = float(data.get("humedad_promedio_pct", 75))
        dsll = int(data.get("dias_sin_lluvia", 5))
        viento = float(data.get("velocidad_viento_ms", 2.0))
        rad = float(data.get("radiacion_solar_kwh", 4.5))
        area = float(data.get("area_sembrada_ha", 2.0))
        dias = int(data.get("dias_desde_siembra", 90))
        dens = float(data.get("densidad_siembra_rel", 1.0))
        fert = int(data.get("nivel_fertilizacion", 1))
        riego = int(data.get("tiene_riego", 0))
        plagas = int(data.get("nivel_control_plagas", 1))

        amplitud = t_max - t_min
        idx_hum = prec / (dsll + 1)
        idx_fert = mat_org * 0.4 + nit / 60 * 0.3 + fos / 50 * 0.3
        score_p = fert / 2 * 0.4 + riego * 0.3 + plagas / 2 * 0.3

        return np.array([[
            cultivo_enc, dpto_enc, suelo_enc, variedad_enc, labranza_enc, es_permanente,
            ph, alt, mat_org, nit, fos, pot,
            temp, t_max, t_min, prec, hum, dsll, viento, rad,
            area, dias, dens, fert, riego, plagas,
            amplitud, idx_hum, idx_fert, score_p,
        ]], dtype=np.float32)

    def _build_lstm_input(self, data: dict):
        """Construye la secuencia temporal para LSTM."""
        datos_clima = data.get("datos_clima", [])
        cultivo = data.get("cultivo", "platano")
        departamento = data.get("departamento", "Caldas")
        variedad = data.get("variedad", "mejorada")

        temp = float(data.get("temp_promedio_c", 20))
        t_max = float(data.get("temp_maxima_c", temp + 5))
        t_min = float(data.get("temp_minima_c", temp - 5))
        prec = float(data.get("precipitacion_mm_90d", 500))
        hum = float(data.get("humedad_promedio_pct", 75))
        dsll = int(data.get("dias_sin_lluvia", 5))
        viento = float(data.get("velocidad_viento_ms", 2.0))
        rad = float(data.get("radiacion_solar_kwh", 4.5))

        seq = []
        for i in range(SEQ_LEN):
            if i < len(datos_clima):
                d = datos_clima[i]
                row = [
                    float(d.get("temp_promedio_c", temp)),
                    float(d.get("temp_maxima_c", t_max)),
                    float(d.get("temp_minima_c", t_min)),
                    float(d.get("precipitacion_mm_90d", prec)),
                    float(d.get("humedad_promedio_pct", hum)),
                    float(d.get("dias_sin_lluvia", dsll)),
                    float(d.get("velocidad_viento_ms", viento)),
                    float(d.get("radiacion_solar_kwh", rad)),
                ]
            else:
                row = [temp, t_max, t_min, prec, hum, dsll, viento, rad]
            seq.append(row)

        seq_arr = np.array(seq, dtype=np.float32)
        scaler_t = MinMaxScaler()
        scaler_t.fit(seq_arr)
        seq_norm = scaler_t.transform(seq_arr)

        cultivo_enc = LabelEncoder().fit(CULTIVOS).transform(
            [cultivo if cultivo in CULTIVOS else "platano"])[0]
        dpto_enc = LabelEncoder().fit(DPTOS).transform(
            [departamento if departamento in DPTOS else "Caldas"])[0]
        variedad_enc = LabelEncoder().fit(VARIEDADES).transform(
            [variedad if variedad in VARIEDADES else "mejorada"])[0]

        static = np.array([[
            cultivo_enc, dpto_enc,
            float(data.get("ph_suelo", 6.0)),
            float(data.get("altitud_msnm", 1500)),
            float(data.get("materia_organica_pct", 3.0)),
            int(data.get("nivel_fertilizacion", 1)),
            int(data.get("tiene_riego", 0)),
            int(data.get("nivel_control_plagas", 1)),
            variedad_enc,
            float(data.get("area_sembrada_ha", 2.0)),
        ]], dtype=np.float32)

        scaler_s = MinMaxScaler()
        scaler_s.fit(static)
        static_norm = scaler_s.transform(static)

        return (
            torch.FloatTensor(seq_norm).unsqueeze(0),
            torch.FloatTensor(static_norm),
        )

    def _calcular_confianza(
        self,
        data: dict,
        tipo_modelo: str,
        pred_xgb: Optional[float],
        pred_lstm: Optional[float],
    ) -> float:
        """
        Calcula la confianza de forma dinámica según:
        1. Completitud de los datos de entrada (más datos = más confianza)
        2. Consistencia entre modelos (si xgb y lstm coinciden = más confianza)
        3. Confianza base por tipo de modelo
        """
        # Confianza base por tipo de modelo
        base = {"xgboost": 75.0, "lstm": 70.0, "ensemble": 80.0}.get(tipo_modelo, 70.0)

        # ── Bonus por completitud de datos (+15 puntos máximo) ──
        campos_criticos = [
            "ph_suelo", "altitud_msnm", "temp_promedio_c",
            "precipitacion_mm_90d", "humedad_promedio_pct",
            "nitrogeno_ppm", "fosforo_ppm", "potasio_meq",
        ]
        campos_opcionales = [
            "materia_organica_pct", "dias_sin_lluvia", "velocidad_viento_ms",
            "radiacion_solar_kwh", "nivel_fertilizacion", "tiene_riego",
            "nivel_control_plagas", "variedad",
        ]
        presentes_criticos = sum(1 for c in campos_criticos if data.get(c) is not None)
        presentes_opcionales = sum(1 for c in campos_opcionales if data.get(c) is not None)
        bonus_completitud = (presentes_criticos / len(campos_criticos)) * 10.0
        bonus_completitud += (presentes_opcionales / len(campos_opcionales)) * 5.0

        # ── Bonus por consistencia entre modelos (+5 puntos) ──
        bonus_consistencia = 0.0
        if pred_xgb is not None and pred_lstm is not None:
            diferencia_pct = abs(pred_xgb - pred_lstm) / max(pred_xgb, pred_lstm, 0.001)
            if diferencia_pct < 0.10:
                bonus_consistencia = 5.0   # modelos muy alineados
            elif diferencia_pct < 0.25:
                bonus_consistencia = 2.5
            else:
                bonus_consistencia = 0.0   # modelos discrepan mucho

        # ── Penalización por datos climáticos serie vacía ──
        penalizacion_serie = 0.0
        if tipo_modelo in ("lstm", "ensemble"):
            n_registros = len(data.get("datos_clima", []))
            if n_registros == 0:
                penalizacion_serie = 8.0   # usando valores estáticos interpolados
            elif n_registros < 15:
                penalizacion_serie = 4.0   # serie incompleta

        confianza = base + bonus_completitud + bonus_consistencia - penalizacion_serie
        return round(max(40.0, min(97.0, confianza)), 1)

    def _calcular_fecha_cosecha(
        self,
        cultivo: str,
        fecha_siembra: str | None,
        dias_desde_siembra: int,
        rendimiento: float,
        temp_promedio: float,
        altitud: float,
    ) -> tuple:
        """
        Estima la fecha de cosecha usando modelo agronómico real:
        - Ciclo base por cultivo (AGRONET / AGROSAVIA Colombia)
        - Ajuste térmico: temperatura afecta velocidad de desarrollo
        - Ajuste altitudinal: a mayor altura, ciclos más largos
        - Ajuste por rendimiento predicho: bajo rendimiento puede indicar
          desarrollo más lento por estrés
        """
        from datetime import datetime, timedelta

        params = CULTIVO_PARAMS.get(cultivo, CULTIVO_PARAMS["platano"])
        ciclo_base = params["ciclo_base"]
        temp_ref = params["temp_ref"]
        alt_ref = params["alt_ref"]
        ajuste_100m = params["ajuste_alt_cada_100m"]
        rend_optimo = params["rend_optimo"]

        # ── Ajuste por temperatura (GDD simplificado) ──
        # Por cada grado por debajo de temp_ref, el cultivo tarda ~2% más
        # Por encima de temp_ref (>3°C), hay estrés por calor: también retarda
        delta_temp = temp_promedio - temp_ref
        if delta_temp < 0:
            # Frío: ralentiza crecimiento linealmente
            ajuste_temp = int(abs(delta_temp) * ciclo_base * 0.018)
        elif delta_temp > 3:
            # Calor excesivo: estrés que también retarda
            ajuste_temp = int((delta_temp - 3) * ciclo_base * 0.01)
        else:
            ajuste_temp = 0  # rango óptimo

        # ── Ajuste por altitud ──
        alt_sobre_ref = max(0, altitud - alt_ref)
        ajuste_alt = int((alt_sobre_ref / 100) * ajuste_100m)

        # ── Ajuste por rendimiento predicho ──
        # Si el rendimiento está por debajo del 60% del óptimo → estrés → ciclo más largo
        ratio_rend = rendimiento / rend_optimo if rend_optimo > 0 else 1.0
        if ratio_rend < 0.6:
            ajuste_rend = int((0.6 - ratio_rend) * ciclo_base * 0.12)
        elif ratio_rend > 1.0:
            # Rendimiento excelente, puede indicar ciclo ligeramente más corto
            ajuste_rend = -int((ratio_rend - 1.0) * ciclo_base * 0.05)
        else:
            ajuste_rend = 0

        # ── Ciclo estimado con límites agronómicos ──
        ciclo_estimado = ciclo_base + ajuste_temp + ajuste_alt + ajuste_rend
        ciclo_estimado = max(params["ciclo_min"], min(params["ciclo_max"], ciclo_estimado))

        dias_restantes = max(0, ciclo_estimado - dias_desde_siembra)

        # ── Calcular fecha absoluta ──
        if fecha_siembra:
            try:
                fs = datetime.fromisoformat(fecha_siembra.replace("Z", "+00:00"))
                fecha_cosecha = fs + timedelta(days=ciclo_estimado)
            except Exception:
                fecha_cosecha = datetime.utcnow() + timedelta(days=dias_restantes)
        else:
            fecha_cosecha = datetime.utcnow() + timedelta(days=dias_restantes)

        logger.info(
            "Cosecha estimada [%s]: ciclo_base=%d ajuste_temp=%+d ajuste_alt=%+d " \
            "ajuste_rend=%+d → ciclo=%d días | restantes=%d",
            cultivo, ciclo_base, ajuste_temp, ajuste_alt, ajuste_rend, ciclo_estimado, dias_restantes
        )
        return fecha_cosecha, dias_restantes

    def _calcular_nivel_riesgo(self, rendimiento: float, cultivo: str) -> NivelRiesgo:
        """
        Calcula el nivel de riesgo como porcentaje del rendimiento óptimo del cultivo.
        Más preciso que usar rangos absolutos: un rendimiento de 10 ton/ha es
        excelente para cacao pero crítico para plátano.
        """
        params = CULTIVO_PARAMS.get(cultivo, CULTIVO_PARAMS["platano"])
        rend_optimo = params["rend_optimo"]
        rend_min = params["rend_min_absoluto"]

        if rend_optimo <= rend_min:
            return NivelRiesgo.MEDIO

        # Porcentaje sobre el rango viable del cultivo
        ratio = (rendimiento - rend_min) / (rend_optimo - rend_min)

        if ratio >= 0.75:
            return NivelRiesgo.BAJO      # ≥ 75% del óptimo
        elif ratio >= 0.50:
            return NivelRiesgo.MEDIO     # 50–75% del óptimo
        elif ratio >= 0.25:
            return NivelRiesgo.ALTO      # 25–50% del óptimo
        return NivelRiesgo.CRITICO       # < 25% del óptimo

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        data = request.datos_agronomicos or {}
        tipo_modelo = request.modelo

        cultivo = data.get("cultivo", "platano")
        if cultivo not in CULTIVOS:
            cultivo = "platano"

        pred_xgb: Optional[float] = None
        pred_lstm: Optional[float] = None
        rendimiento: float = 1.0

        # ── XGBoost — modelo específico del cultivo ──
        if cultivo in self._xgb_models:
            try:
                features = self._encode_features(data)
                pred_xgb = float(self._xgb_models[cultivo].predict(features)[0])
                logger.info("XGBoost[%s] predicción: %.3f ton/ha", cultivo, pred_xgb)
            except Exception as e:
                logger.error("Error XGBoost[%s] predict: %s", cultivo, e)

        # ── LSTM ──
        if self._lstm_model is not None:
            try:
                x_seq, x_static = self._build_lstm_input(data)
                with torch.no_grad():
                    pred_lstm = float(self._lstm_model(x_seq, x_static).item())
                logger.info("LSTM predicción: %.3f ton/ha", pred_lstm)
            except Exception as e:
                logger.error("Error LSTM predict: %s", e)

        # ── Ensemble / selección ──
        w_xgb = 0.95
        w_lstm = 0.05
        if self._ensemble_config:
            w_xgb = self._ensemble_config.get("peso_xgboost", 0.95)
            w_lstm = self._ensemble_config.get("peso_lstm", 0.05)

        if tipo_modelo == TipoModelo.XGBOOST and pred_xgb is not None:
            rendimiento = pred_xgb
            confianza = self._calcular_confianza(data, "xgboost", pred_xgb, None)
        elif tipo_modelo == TipoModelo.LSTM and pred_lstm is not None:
            rendimiento = pred_lstm
            confianza = self._calcular_confianza(data, "lstm", None, pred_lstm)
        elif pred_xgb is not None and pred_lstm is not None:
            rendimiento = w_xgb * pred_xgb + w_lstm * pred_lstm
            confianza = self._calcular_confianza(data, "ensemble", pred_xgb, pred_lstm)
        elif pred_xgb is not None:
            rendimiento = pred_xgb
            confianza = self._calcular_confianza(data, "xgboost", pred_xgb, None)
        elif pred_lstm is not None:
            rendimiento = pred_lstm
            confianza = self._calcular_confianza(data, "lstm", None, pred_lstm)
        else:
            # Sin modelo disponible — usar estimación por defecto del cultivo
            _cultivo_key = data.get("cultivo", "platano") if cultivo not in CULTIVO_PARAMS else cultivo
            params = CULTIVO_PARAMS.get(_cultivo_key, CULTIVO_PARAMS["platano"])
            rendimiento = params["rend_optimo"] * 0.6  # 60% del óptimo como estimación conservadora
            confianza = 40.0
            logger.warning("Sin modelo disponible para %s, usando default conservador: %.1f", cultivo, rendimiento)

        rendimiento = max(0.1, rendimiento)

        # ── Intervalo de confianza basado en métricas reales ──
        metricas_cultivo = (
            self._xgb_metrics.get("metricas_por_cultivo", {}).get(cultivo, {})
        )
        rmse_pct = metricas_cultivo.get("rmse_pct", 5.0) / 100
        inferior = round(max(0.1, rendimiento * (1 - rmse_pct * 2)), 3)
        superior = round(rendimiento * (1 + rmse_pct * 2), 3)

        nivel_riesgo = self._calcular_nivel_riesgo(rendimiento, cultivo)

        # Feature importance del modelo específico
        fi_cultivo = (
            self._xgb_metrics.get("metricas_por_cultivo", {})
            .get(cultivo, {})
            .get("feature_importance", {})
        )
        if not fi_cultivo:
            fi_cultivo = self._xgb_metrics.get("feature_importance", {})
        top_features = dict(list(fi_cultivo.items())[:5]) if fi_cultivo else {}

        # ── Fecha de cosecha ──
        fecha_cosecha_est, dias_para_cosecha = self._calcular_fecha_cosecha(
            cultivo=cultivo,
            fecha_siembra=data.get("fecha_siembra"),
            dias_desde_siembra=int(data.get("dias_desde_siembra", 90)),
            rendimiento=rendimiento,
            temp_promedio=float(data.get("temp_promedio_c", 25.0)),
            altitud=float(data.get("altitud_msnm", 500.0)),
        )

        return PredictionResponse(
            parcela_id=request.parcela_id,
            cultivo_parcela_id=request.cultivo_parcela_id,
            tipo_cultivo_id=request.tipo_cultivo_id,
            version_modelo=self.MODEL_VERSION,
            tipo_modelo=tipo_modelo.value,
            rendimiento_predicho_ton=round(rendimiento, 3),
            puntaje_confianza=confianza,
            intervalo_conf_inferior=inferior,
            intervalo_conf_superior=superior,
            nivel_riesgo=nivel_riesgo,
            factores_riesgo={
                "cultivo": cultivo,
                "nivel": nivel_riesgo.value,
                "rendimiento_esperado_min": inferior,
                "rendimiento_esperado_max": superior,
                "dias_desde_siembra": int(data.get("dias_desde_siembra", 90)),
                "modelo_usado": "xgboost" if pred_xgb is not None else "lstm" if pred_lstm is not None else "default",
            },
            datos_clima_usados={
                "temp_promedio": data.get("temp_promedio_c"),
                "temp_maxima": data.get("temp_maxima_c"),
                "temp_minima": data.get("temp_minima_c"),
                "precipitacion_mm_90d": data.get("precipitacion_mm_90d"),
                "humedad_pct": data.get("humedad_promedio_pct"),
                "altitud_msnm": data.get("altitud_msnm"),
                "dias_sin_lluvia": data.get("dias_sin_lluvia"),
                "registros_clima_usados": len(data.get("datos_clima", [])),
            },
            importancia_features=top_features,
            fecha_prediccion=datetime.utcnow(),
            fecha_cosecha_estimada=fecha_cosecha_est,
            dias_para_cosecha=dias_para_cosecha,
        )

    def get_available_models(self) -> list[ModelInfo]:
        w_xgb = self._ensemble_config.get("peso_xgboost", 0.95) if self._ensemble_config else 0.95
        w_lstm = self._ensemble_config.get("peso_lstm", 0.05) if self._ensemble_config else 0.05
        return [
            ModelInfo(
                nombre="XGBoost — Predicción Base",
                tipo=TipoModelo.XGBOOST,
                version=self.MODEL_VERSION,
                estado="cargado" if self._models_loaded["xgboost"] else "no disponible",
                descripcion="Gradient boosting para features estáticas.",
                metricas=self._xgb_metrics.get("test", {}),
            ),
            ModelInfo(
                nombre="LSTM — Series Temporales",
                tipo=TipoModelo.LSTM,
                version=self.MODEL_VERSION,
                estado="cargado" if self._models_loaded["lstm"] else "no disponible",
                descripcion="Red LSTM + Attention para series climáticas de 30 días.",
                metricas=self._lstm_metrics.get("test", {}),
            ),
            ModelInfo(
                nombre="Ensemble — XGBoost + LSTM",
                tipo=TipoModelo.ENSEMBLE,
                version=self.MODEL_VERSION,
                estado="cargado" if self._models_loaded["ensemble"] else "no disponible",
                descripcion=f"Combinación ponderada: XGBoost {int(w_xgb*100)}% + LSTM {int(w_lstm*100)}%.",
                metricas=self._ensemble_config.get("test", {}) if self._ensemble_config else {},
            ),
        ]

    @property
    def models_loaded_count(self) -> int:
        return sum(1 for v in self._models_loaded.values() if v)

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time