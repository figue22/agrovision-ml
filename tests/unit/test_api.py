import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app


@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "agrovision-ml"
    assert "version" in data
    assert "models_loaded" in data
    assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_models_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/models")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["modelos"]) == 3
    tipos = [m["tipo"] for m in data["modelos"]]
    assert "xgboost" in tipos
    assert "lstm" in tipos
    assert "ensemble" in tipos


@pytest.mark.asyncio
async def test_predict_stub():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={
            "parcela_id": "550e8400-e29b-41d4-a716-446655440000",
            "cultivo_parcela_id": "660e8400-e29b-41d4-a716-446655440001",
            "tipo_cultivo_id": "770e8400-e29b-41d4-a716-446655440002",
            "modelo": "ensemble",
        })

    assert response.status_code == 200
    data = response.json()
    assert "rendimiento_predicho_ton" in data
    assert "puntaje_confianza" in data
    assert "nivel_riesgo" in data
    assert data["tipo_modelo"] == "ensemble"
    assert data["rendimiento_predicho_ton"] > 0


@pytest.mark.asyncio
async def test_predict_validation_error():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={
            "parcela_id": "",
            "cultivo_parcela_id": "",
        })

    assert response.status_code == 422
