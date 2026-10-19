
import threading
import time
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import api
from app.database import get_db
from app.db_models import ClaimRun
from app.schemas import AgentResult


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """Create an isolated in-memory database for each test."""
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

    # The API lifespan creates tables using api.engine.
    monkeypatch.setattr(api, "engine", test_engine)

    def override_get_db():
        db = test_session()
        try:
            yield db
        finally:
            db.close()

    api.app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(api.app) as test_client:
            yield test_client
    finally:
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

    TestSession = sessionmaker(
        bind=api.engine,
        autoflush=False,
        expire_on_commit=False,
    )
    db = TestSession()

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


def test_provider_concurrency_is_limited_to_three(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0
    start_barrier = threading.Barrier(3)

    class FakeAgent:
        def classify(self, answer):
            nonlocal active_calls, max_active_calls

            with lock:
                active_calls += 1
                max_active_calls = max(
                    max_active_calls,
                    active_calls,
                )

            try:
                # The first three requests can overlap.
                start_barrier.wait(timeout=5)
                time.sleep(0.1)

                return AgentResult(
                    answer_id=answer["answer_id"],
                    claims=[],
                )
            finally:
                with lock:
                    active_calls -= 1

    monkeypatch.setattr(
        api,
        "GeminiClaimAgent",
        lambda: FakeAgent(),
    )

    payloads = [
        {
            "answer_id": f"CONCURRENT-{i}",
            "brand_id": "cleartrail",
            "question": "Tell me about the product.",
            "ai_answer": "Concurrency test answer.",
        }
        for i in range(6)
    ]

    results = []
    errors = []

    def submit(payload):
        try:
            response = client.post("/claims", json=payload)
            results.append(response.status_code)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=submit, args=(payload,))
        for payload in payloads
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert sorted(results) == [200] * 6
    assert max_active_calls == 3
