import io
import json

import pytest

import server


class FakeRetriever:
    def __init__(self, best_score):
        self.best_score = best_score
        self.queries = []

    def search(self, query, k=3):
        self.queries.append(query)
        return ["Static clipping uses one fixed threshold."], self.best_score


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "current_paper", None)
    return server.app.test_client()


def load_paper(monkeypatch, best_score=0.8):
    retriever = FakeRetriever(best_score)
    paper = server.Paper("paper.pdf", "unused.pdf", retriever, ["chunk"], {})
    monkeypatch.setattr(server, "current_paper", paper)
    return retriever


def events_of(response):
    return [json.loads(line) for line in response.get_data(as_text=True).splitlines()]


def test_health_reports_that_no_paper_is_loaded(client):
    data = client.get("/api/health").get_json()
    assert data["app"] == "ieee-paper-assistant"
    assert data["paper"] is None


def test_upload_rejects_a_file_that_is_not_a_pdf(client):
    response = client.post("/api/upload", data={"file": (io.BytesIO(b"hello"), "notes.pdf")},
                           content_type="multipart/form-data")
    assert response.status_code == 400
    assert "not a PDF" in response.get_json()["error"]


def test_other_websites_are_rejected_and_the_github_pages_site_is_allowed(client):
    assert client.get("/api/health", headers={"Origin": "https://evil.example"}).status_code == 403
    response = client.get("/api/health", headers={"Origin": "https://tianxinxie123.github.io"})
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "https://tianxinxie123.github.io"
    assert client.get("/api/health", headers={"Origin": "http://localhost:5500"}).status_code == 200


def test_ask_needs_a_paper(client):
    assert client.post("/api/ask", json={"question": "What is it?"}).status_code == 409


def test_ask_streams_sources_then_the_answer(client, monkeypatch):
    load_paper(monkeypatch)

    def fake_llm(query, chunks, provider, model, history, on_text):
        on_text("Fixed ")
        on_text("threshold [Source 1].")
        return "Fixed threshold [Source 1]."

    monkeypatch.setattr(server, "get_llm_response", fake_llm)
    events = events_of(client.post("/api/ask", json={"question": "What does static clipping use?"}))
    types = [e["type"] for e in events]
    assert types[-4:] == ["status", "text", "text", "done"]
    assert {"type": "sources", "sources": ["Static clipping uses one fixed threshold."]} in events
    assert "".join(e["text"] for e in events if e["type"] == "text") == "Fixed threshold [Source 1]."


def test_off_topic_question_gets_a_notice_without_calling_the_llm(client, monkeypatch):
    load_paper(monkeypatch, best_score=0.0)
    monkeypatch.setattr(server, "get_llm_response", lambda *a, **kw: pytest.fail("LLM was called"))
    events = events_of(client.post("/api/ask", json={"question": "Best pizza in Waterloo?"}))
    assert events[-1]["type"] == "notice"


def test_force_prefix_skips_the_relevance_check(client, monkeypatch):
    retriever = load_paper(monkeypatch, best_score=0.0)
    monkeypatch.setattr(server, "get_llm_response", lambda query, *a, on_text, **kw: on_text("ok"))
    events = events_of(client.post("/api/ask", json={"question": "! Best pizza in Waterloo?"}))
    assert retriever.queries == ["Best pizza in Waterloo?"]
    assert events[-1]["type"] == "done"


def test_llm_error_becomes_an_error_event(client, monkeypatch):
    load_paper(monkeypatch)

    def failing_llm(*args, **kwargs):
        raise server.LLMError("Claude rate limit reached.")

    monkeypatch.setattr(server, "get_llm_response", failing_llm)
    events = events_of(client.post("/api/ask", json={"question": "What does static clipping use?"}))
    assert events[-1] == {"type": "error", "text": "Claude rate limit reached."}
