import httpx
import pytest

from trajectory.backend import InferenceBackend, VLLMBackend
from trajectory.datatypes import ModelRequest, ModelResponse

pytestmark = pytest.mark.asyncio


async def test_inference_backend_is_abstract():
    with pytest.raises(TypeError):
        InferenceBackend()


async def test_vllm_backend_generate_forwards_and_parses(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured["client_init_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "test response"},
                            "finish_reason": "stop",
                            "token_ids": [101, 102],
                            "logprobs": {
                                "content": [
                                    {"token": "test", "logprob": -0.5},
                                    {"token": " response", "logprob": -0.3},
                                ]
                            },
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("trajectory.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm", actual_model="Qwen3-4B")
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hello"}],
        generation_params={"temperature": 0.7, "max_tokens": 128},
    )

    resp = await backend.generate(req)

    assert captured["url"] == "http://fake-vllm/v1/chat/completions"
    assert isinstance(resp, ModelResponse)
    assert resp.content == "test response"
    assert resp.finish_reason == "stop"
    assert resp.token_ids == [101, 102]
    assert resp.logprobs == [-0.5, -0.3]
    assert captured["json"] == {
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        "max_tokens": 128,
        "logprobs": True,
        "return_token_ids": True,
        "model": "Qwen3-4B",
    }


async def test_vllm_backend_forces_logprobs_true(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["json"] = json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("trajectory.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm")
    req = ModelRequest(
        request_id="r1",
        agent_role="searcher",
        messages=[{"role": "user", "content": "x"}],
        generation_params={"logprobs": False},
    )
    await backend.generate(req)

    assert captured["json"]["logprobs"] is True
    assert captured["json"]["return_token_ids"] is True
    assert captured["json"]["model"] == "searcher"


async def test_vllm_backend_actual_model_overrides_request_model(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["json"] = json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("trajectory.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm", actual_model="real-model-name")
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "x"}],
        generation_params={"model": "placeholder-model"},
    )
    await backend.generate(req)

    assert captured["json"]["model"] == "real-model-name"


async def test_vllm_backend_uses_backend_url_override_and_drops_reserved_key(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("trajectory.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://default-vllm")
    req = ModelRequest(
        request_id="r1",
        agent_role="searcher",
        messages=[{"role": "user", "content": "x"}],
        generation_params={"temperature": 0.3, "_backend_url": "http://role-vllm/"},
    )
    await backend.generate(req)

    assert captured["url"] == "http://role-vllm/v1/chat/completions"
    assert captured["json"] == {
        "messages": [{"role": "user", "content": "x"}],
        "temperature": 0.3,
        "logprobs": True,
        "return_token_ids": True,
        "model": "searcher",
    }


async def test_vllm_backend_raises_value_error_for_malformed_choices(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            return httpx.Response(
                200,
                json={"choices": []},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("trajectory.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm")
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "x"}],
        generation_params={},
    )

    with pytest.raises(ValueError, match="malformed response"):
        await backend.generate(req)


async def test_vllm_backend_logprobs_keeps_only_finite_numbers(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {"token": "a", "logprob": -0.5},
                                {"token": "b", "logprob": 2},
                                {"token": "c", "logprob": "bad"},
                                {"token": "d", "logprob": None},
                                {"token": "e", "logprob": float("inf")},
                                {"token": "f", "logprob": float("-inf")},
                                {"token": "g", "logprob": float("nan")},
                                {"token": "h", "logprob": True},
                            ]
                        },
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr("trajectory.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm")
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "x"}],
        generation_params={},
    )
    resp = await backend.generate(req)

    assert resp.logprobs == [-0.5, 2.0]
