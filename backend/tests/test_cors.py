from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_cors_preflight():
    for origin in [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]:
        response = client.options(
            "/api/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == origin
        assert response.headers.get("access-control-allow-credentials") == "true"
