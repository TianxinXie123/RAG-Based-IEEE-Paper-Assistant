import io
import json
from collections import OrderedDict

import pytest

import server

ALICE = {"X-Visitor-Id": "alice-0001"}
BOB = {"X-Visitor-Id": "bob-0002"}


class FakeRetriever:
    def __init__(self, best_score):
        self.best_score = best_score
        self.queries = []

    def search(self, query, k=3):
        self.queries.append(query)
        return ["Static clipping uses one fixed threshold."], self.best_score


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "papers", OrderedDict())
    monkeypatch.setattr(server, "limiters", {})
    return server.app.test_client()


def load_paper(best_score=0.8, visitor="default", name="paper.pdf"):
    retriever = FakeRetriever(best_score)
    server.set_paper(visitor, server.Paper(name, "unused.pdf", retriever, ["chunk"], {}))
    return retriever


def answer_with(monkeypatch, text):
    monkeypatch.setattr(server, "get_llm_response", lambda query, *a, on_text, **kw: on_text(text))


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


def test_the_hugging_face_space_page_is_allowed(client, monkeypatch):
    space = {"Origin": "https://tianxinxie123-ieee-paper-assistant.hf.space"}
    assert client.get("/api/health", headers=space).status_code == 403
    monkeypatch.setenv("SPACE_HOST", "tianxinxie123-ieee-paper-assistant.hf.space")
    assert client.get("/api/health", headers=space).status_code == 200
    assert client.get("/api/health", headers={"Origin": "https://other-space.hf.space"}).status_code == 403


def test_the_visitor_header_is_allowed_by_cors(client):
    response = client.options("/api/ask", headers={"Origin": "https://tianxinxie123.github.io"})
    assert "X-Visitor-Id" in response.headers["Access-Control-Allow-Headers"]


def test_each_visitor_has_their_own_paper(client):
    load_paper(visitor="alice-0001", name="alice.pdf")
    assert client.get("/api/health", headers=ALICE).get_json()["paper"]["name"] == "alice.pdf"
    assert client.get("/api/health", headers=BOB).get_json()["paper"] is None
    assert client.post("/api/ask", headers=BOB, json={"question": "What is it?"}).status_code == 409
    # An id that is not a plain random string counts as no id
    assert client.get("/api/health", headers={"X-Visitor-Id": "../x"}).get_json()["paper"] is None


def test_the_least_recently_used_paper_is_dropped(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_PAPERS", 2)
    for visitor in ("first-0001", "second-002", "third-0003"):
        load_paper(visitor=visitor)
    assert list(server.papers) == ["second-002", "third-0003"]


def test_questions_are_limited_per_visitor(client, monkeypatch):
    monkeypatch.setattr(server, "limiters", {"ask": (server.RateLimiter(1, 3600), server.RateLimiter(100, 86400))})
    load_paper(visitor="alice-0001")
    load_paper(visitor="bob-0002")
    answer_with(monkeypatch, "ok")
    assert client.post("/api/ask", headers=ALICE, json={"question": "What is it?"}).status_code == 200
    response = client.post("/api/ask", headers=ALICE, json={"question": "And then?"})
    assert response.status_code == 429
    assert "1 questions per hour" in response.get_json()["error"]
    assert client.post("/api/ask", headers=BOB, json={"question": "What is it?"}).status_code == 200


def test_questions_are_limited_per_day_for_all_visitors(client, monkeypatch):
    monkeypatch.setattr(server, "limiters", {"ask": (server.RateLimiter(10, 3600), server.RateLimiter(1, 86400))})
    load_paper(visitor="alice-0001")
    load_paper(visitor="bob-0002")
    answer_with(monkeypatch, "ok")
    assert client.post("/api/ask", headers=ALICE, json={"question": "What is it?"}).status_code == 200
    response = client.post("/api/ask", headers=BOB, json={"question": "What is it?"})
    assert response.status_code == 429
    assert "daily limit" in response.get_json()["error"]


def test_rejected_requests_do_not_count_toward_the_limit(client, monkeypatch):
    monkeypatch.setattr(server, "limiters", {"ask": (server.RateLimiter(1, 3600), server.RateLimiter(100, 86400))})
    load_paper()
    answer_with(monkeypatch, "ok")
    assert client.post("/api/ask", json={"question": ""}).status_code == 400
    assert client.post("/api/ask", json={"question": "What is it?"}).status_code == 200


def test_rate_limiter_forgets_requests_older_than_its_window(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(server.time, "monotonic", lambda: now[0])
    limiter = server.RateLimiter(2, 60)
    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    now[0] += 60
    assert limiter.allow("a")


def test_ask_needs_a_paper(client):
    assert client.post("/api/ask", json={"question": "What is it?"}).status_code == 409


def test_ask_streams_sources_then_the_answer(client, monkeypatch):
    load_paper()

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
    load_paper(best_score=0.0)
    monkeypatch.setattr(server, "get_llm_response", lambda *a, **kw: pytest.fail("LLM was called"))
    events = events_of(client.post("/api/ask", json={"question": "Best pizza in Waterloo?"}))
    assert events[-1]["type"] == "notice"


def test_force_prefix_skips_the_relevance_check(client, monkeypatch):
    retriever = load_paper(best_score=0.0)
    answer_with(monkeypatch, "ok")
    events = events_of(client.post("/api/ask", json={"question": "! Best pizza in Waterloo?"}))
    assert retriever.queries == ["Best pizza in Waterloo?"]
    assert events[-1]["type"] == "done"


def test_llm_error_becomes_an_error_event(client, monkeypatch):
    load_paper()

    def failing_llm(*args, **kwargs):
        raise server.LLMError("Claude rate limit reached.")

    monkeypatch.setattr(server, "get_llm_response", failing_llm)
    events = events_of(client.post("/api/ask", json={"question": "What does static clipping use?"}))
    assert events[-1] == {"type": "error", "text": "Claude rate limit reached."}
