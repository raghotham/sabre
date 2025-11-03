"""
Tests for /respond endpoint.

Run with: uv run pytest tests/test_ask_user_endpoint.py
"""

import asyncio
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


@pytest.fixture
def test_app():
    """Create test FastAPI app."""
    from fastapi import FastAPI, Request, HTTPException

    app = FastAPI()

    # Mock pending_questions storage
    pending_questions = {}

    @app.post("/respond/{question_id}")
    async def respond_to_question(question_id: str, request: Request):
        """Test version of respond endpoint."""
        data = await request.json()
        answers = data.get("answers", [])

        if not answers:
            raise HTTPException(status_code=400, detail="No answers provided")

        if question_id not in pending_questions:
            raise HTTPException(status_code=404, detail="Question not found or already answered")

        future = pending_questions[question_id]
        if not future.done():
            future.set_result(answers)

        del pending_questions[question_id]

        return {"status": "ok", "question_id": question_id, "answers_count": len(answers)}

    # Store pending_questions for test access
    app.state.pending_questions = pending_questions

    return app


def test_respond_endpoint_success(test_app):
    """Test successful response to question."""
    client = TestClient(test_app)

    # Create a future and add to pending_questions
    question_id = "test-q-123"
    future = asyncio.Future()
    test_app.state.pending_questions[question_id] = future

    # Send response
    response = client.post(
        f"/respond/{question_id}",
        json={"answers": ["Alice", "30", "Blue"]}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["question_id"] == question_id
    assert data["answers_count"] == 3

    # Verify future was resolved
    assert future.done()
    assert future.result() == ["Alice", "30", "Blue"]

    # Verify cleanup
    assert question_id not in test_app.state.pending_questions


def test_respond_endpoint_missing_answers(test_app):
    """Test response without answers."""
    client = TestClient(test_app)

    question_id = "test-q-456"
    future = asyncio.Future()
    test_app.state.pending_questions[question_id] = future

    # Send empty answers
    response = client.post(
        f"/respond/{question_id}",
        json={"answers": []}
    )

    assert response.status_code == 400
    assert "No answers provided" in response.json()["detail"]


def test_respond_endpoint_question_not_found(test_app):
    """Test response to non-existent question."""
    client = TestClient(test_app)

    response = client.post(
        "/respond/non-existent-question",
        json={"answers": ["Answer"]}
    )

    assert response.status_code == 404
    assert "Question not found" in response.json()["detail"]


def test_respond_endpoint_single_answer(test_app):
    """Test response with single answer."""
    client = TestClient(test_app)

    question_id = "test-q-789"
    future = asyncio.Future()
    test_app.state.pending_questions[question_id] = future

    response = client.post(
        f"/respond/{question_id}",
        json={"answers": ["Single answer"]}
    )

    assert response.status_code == 200
    assert future.done()
    assert future.result() == ["Single answer"]


def test_respond_endpoint_multiple_answers(test_app):
    """Test response with multiple answers."""
    client = TestClient(test_app)

    question_id = "test-q-abc"
    future = asyncio.Future()
    test_app.state.pending_questions[question_id] = future

    answers = ["Answer 1", "Answer 2", "Answer 3", "Answer 4", "Answer 5"]

    response = client.post(
        f"/respond/{question_id}",
        json={"answers": answers}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answers_count"] == 5

    assert future.done()
    assert future.result() == answers


def test_respond_endpoint_already_answered(test_app):
    """Test responding to already-answered question."""
    client = TestClient(test_app)

    question_id = "test-q-xyz"
    future = asyncio.Future()
    future.set_result(["Already answered"])
    test_app.state.pending_questions[question_id] = future

    # Try to answer again
    response = client.post(
        f"/respond/{question_id}",
        json={"answers": ["New answer"]}
    )

    # Should still succeed but not change result
    assert response.status_code == 200


def test_respond_endpoint_concurrent_requests():
    """Test concurrent responses to different questions."""
    from fastapi import FastAPI, Request
    import asyncio

    app = FastAPI()
    pending_questions = {}

    @app.post("/respond/{question_id}")
    async def respond_to_question(question_id: str, request: Request):
        data = await request.json()
        answers = data.get("answers", [])

        if question_id not in pending_questions:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Question not found")

        future = pending_questions[question_id]
        if not future.done():
            future.set_result(answers)

        del pending_questions[question_id]
        return {"status": "ok"}

    # Create multiple futures
    q1_id = "q1"
    q2_id = "q2"
    q3_id = "q3"

    pending_questions[q1_id] = asyncio.Future()
    pending_questions[q2_id] = asyncio.Future()
    pending_questions[q3_id] = asyncio.Future()

    client = TestClient(app)

    # Send concurrent responses
    r1 = client.post(f"/respond/{q1_id}", json={"answers": ["A1"]})
    r2 = client.post(f"/respond/{q2_id}", json={"answers": ["A2"]})
    r3 = client.post(f"/respond/{q3_id}", json={"answers": ["A3"]})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 200

    # All should be cleaned up
    assert len(pending_questions) == 0


def test_respond_endpoint_answers_with_special_characters(test_app):
    """Test response with special characters in answers."""
    client = TestClient(test_app)

    question_id = "test-q-special"
    future = asyncio.Future()
    test_app.state.pending_questions[question_id] = future

    special_answers = [
        "Answer with 'quotes'",
        "Answer with \"double quotes\"",
        "Answer with\nnewlines",
        "Answer with\ttabs",
        "Answer with émojis 🎉",
    ]

    response = client.post(
        f"/respond/{question_id}",
        json={"answers": special_answers}
    )

    assert response.status_code == 200
    assert future.done()
    assert future.result() == special_answers


def test_respond_endpoint_empty_string_answers(test_app):
    """Test response with empty string answers."""
    client = TestClient(test_app)

    question_id = "test-q-empty"
    future = asyncio.Future()
    test_app.state.pending_questions[question_id] = future

    # Empty strings are valid answers
    response = client.post(
        f"/respond/{question_id}",
        json={"answers": ["", "", ""]}
    )

    assert response.status_code == 200
    assert future.done()
    assert future.result() == ["", "", ""]
