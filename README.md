# 🤖 AgroVision ML Service - Predicción de Rendimiento de Cultivos

> Microservicio de Machine Learning con FastAPI para predicción de rendimiento agrícola usando XGBoost y LSTM.

## Descripción

Servicio independiente en Python que implementa modelos de machine learning para predecir rendimiento de cultivos basándose en datos meteorológicos históricos, características del suelo y prácticas agrícolas. Comunica con el backend NestJS vía gRPC. Los resultados se persisten en la tabla `predicciones` y alimentan la generación de `recomendaciones`.

## Stack Tecnológico

### Predicción ML
| Tecnología | Versión | Propósito |
|---|---|---|
| Python | 3.11 | Lenguaje de programación |
| FastAPI | 0.109.0 | Framework API async |
| XGBoost | 2.0.3 | Modelo predicción base |
| TensorFlow | 2.15.0 | LSTM y deep learning |
| Scikit-learn | 1.4.0 | Preprocesamiento ML |
| Pandas | 2.1.4 | Manipulación de datos |
| NumPy | 1.26.3 | Operaciones numéricas |
| MLflow | 2.10.0 | Tracking y versionado de modelos |
| Pydantic | 2.5.3 | Validación de datos |

## Estructura del Proyecto

```
src/
├── api/
│   ├── routes/                 # Endpoints FastAPI
│   │   ├── predictions.py      # POST /predict, GET /predictions
│   │   ├── training.py         # POST /train, GET /models
│   │   └── health.py           # GET /health
│   └── middleware/              # Auth, logging, error handling
├── models/
│   ├── xgboost/                # Modelo XGBoost para predicción base
│   ├── lstm/                   # Modelo LSTM para series temporales
│   └── ensemble/               # Combinación de modelos
├── services/
│   ├── prediction_service.py   # Lógica de predicción
│   └── training_service.py     # Entrenamiento y re-entrenamiento
├── features/                   # Feature engineering pipeline
├── grpc/                       # gRPC server (proto contracts con backend)
├── tracking/                   # MLflow experiment tracking
├── schemas/                    # Pydantic models para validación
├── utils/                      # Helpers
└── config/                     # Configuración y settings

data/
├── raw/                        # Datos crudos (CSV, Excel)
├── processed/                  # Datos preprocesados
└── models/                     # Modelos serializados (.joblib, .h5)

protos/                         # Archivos .proto para gRPC

notebooks/                      # Jupyter notebooks de exploración
scripts/                        # Scripts de entrenamiento y ETL
tests/
├── unit/
└── integration/
docs/
```

## Interacción con Base de Datos (ER v3)

El servicio ML lee y escribe en las siguientes tablas de PostgreSQL:

| Tabla | Uso |
|---|---|
| `parcelas` | Lectura: ubicación, tipo suelo, pH, altitud |
| `tipos_cultivo` | Lectura: parámetros óptimos (temp, altitud, pH, req. agua) |
| `cultivos_parcela` | Lectura: cultivo activo, área sembrada, temporada; Escritura: rendimiento esperado, fecha cosecha esperada |
| `datos_climaticos` | Lectura: series temporales meteorológicas por parcela (temp, precipitación, humedad, viento, UV) |
| `actividades` | Lectura: historial de prácticas agrícolas (fertilización, riego, etc.) |
| `insumos_actividad` | Lectura: insumos aplicados por actividad |
| `predicciones` | Escritura: rendimiento predicho, confianza, intervalo 95%, riesgo, factores de riesgo JSONB, importancia features JSONB |

## Endpoints API

### Predicciones
| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/predict` | Generar predicción de rendimiento |
| `GET` | `/predictions/{parcela_id}` | Historial de predicciones por parcela |
| `GET` | `/models` | Listar modelos disponibles |
| `GET` | `/health` | Health check del servicio |

### Entrenamiento
| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/train` | Iniciar entrenamiento de modelo |
| `GET` | `/train/status/{job_id}` | Estado del entrenamiento |

## Modelos ML

### XGBoost - Predicción Base
- **Input**: Features de suelo (tipo, pH, altitud), clima (series temporales de `datos_climaticos`), cultivo (parámetros de `tipos_cultivo`), prácticas agrícolas (`actividades` + `insumos_actividad`)
- **Output**: Rendimiento estimado (ton/ha), factores de riesgo, importancia de features
- **Métrica objetivo**: RMSE < 15% del rendimiento promedio

### LSTM - Series Temporales Climáticas
- **Input**: Secuencia temporal de datos meteorológicos (90 días de `datos_climaticos`)
- **Output**: Componente climático del rendimiento
- **Ventana**: 90 días de datos históricos

### Ensemble
- Combinación ponderada de XGBoost (60%) y LSTM (40%)
- Score de confianza (0-100) basado en calidad de datos de entrada
- Intervalo de confianza al 95% (inferior y superior)
- Nivel de riesgo: bajo, medio, alto, crítico

## Variables de Entorno

```env
# General
APP_ENV=development
APP_PORT=8000

# Base de datos (lectura de parcelas, cultivos, clima, actividades)
DATABASE_URL=postgresql://user:password@localhost:5432/agrovision

# OpenAI (no usado directamente, solo si se integra explicabilidad)
OPENAI_API_KEY=your-api-key

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=agrovision-predictions

# Modelos
MODEL_PATH=./data/models
```

## Instalación y Ejecución

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env

# Desarrollo
uvicorn src.main:app --reload --port 8000

# Producción
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# Tests
pytest tests/
pytest tests/ --cov=src --cov-report=html

# MLflow UI (tracking de experimentos)
mlflow ui --port 5000
```

## Documentación API

Documentación automática disponible en:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Métricas de Rendimiento

| Métrica | Objetivo |
|---|---|
| RMSE predicción | < 15% rendimiento promedio |
| Latencia predicción (p95) | < 500ms |
| Re-entrenamiento | Trimestral |

## Contribución

1. Crear branch desde `develop`: `git checkout -b feature/nombre-feature`
2. Commits con convención: `feat:`, `fix:`, `docs:`, `refactor:`
3. Pull Request hacia `develop`

## Licencia

Proyecto privado - AgroVision © 2026
