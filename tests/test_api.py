
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import api
from app.database import Base, get_db
from app.db_models import ClaimRun
from app.schemas import AgentResult


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_session = sessionmaker(
        bind=test_engine,
        autoflush=False,
        expire_on_commit=False,
    )

    monkeypatch.setattr(api, "engine", test_engine)

    def override_get_db():
        db = test_session()
        try:
            yield db
        finally:
            db.close()

    api.app.dependency_overrides[get_db] = override_get_db

    with TestClient(api.app) as test_client:
        yield test_client

    api.app.dependency_overrides.clear()
    test_engine.dispose()


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_and_retrieve_claims(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAgent:
        def classify(self, answer):
            return AgentResult(
                answer_id=answer["answer_id"],
                claims=[],
            )

    monkeypatch.setattr(
        api,
        "GeminiClaimAgent",
        lambda: FakeAgent(),
    )

    payload = {
        "answer_id": "TEST-001",
        "brand_id": "cleartrail",
        "question": "Tell me about the product.",
        "ai_answer": "This is a test answer.",
    }

    created = client.post("/claims", json=payload)

    assert created.status_code == 200
    assert created.json() == {
        "answer_id": "TEST-001",
        "claims": [],
    }

    retrieved = client.get("/claims/TEST-001")

    assert retrieved.status_code == 200
    assert retrieved.json() == created.json()


def test_missing_claim_result_returns_404(
    client: TestClient,
) -> None:
    response = client.get("/claims/DOES-NOT-EXIST")

    assert response.status_code == 404


def test_agent_failure_is_recorded_safely(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAgent:
        def classify(self, answer):
            raise api.AgentError("simulated provider failure")

    monkeypatch.setattr(
        api,
        "GeminiClaimAgent",
        lambda: FakeAgent(),
    )

    payload = {
        "answer_id": "TEST-FAIL-001",
        "brand_id": "cleartrail",
        "question": "Tell me about the product.",
        "ai_answer": "This is a test answer.",
    }

    response = client.post("/claims", json=payload)

    assert response.status_code == 502
    assert response.json()["detail"] == "Claim verification failed"
    assert "simulated provider failure" not in response.text

    db = next(api.app.dependency_overrides[get_db]())
    try:
        run = db.scalar(
            select(ClaimRun).where(
                ClaimRun.answer_id == "TEST-FAIL-001"
            )
        )
        assert run is not None
        assert run.status == "failed"
        assert "simulated provider failure" in run.error_message
    finally:
        db.close()
