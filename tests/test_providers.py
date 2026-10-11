import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import httpx
import numpy as np
import pytest
from groq import Groq
from pydantic import SecretStr
from qdrant_client import QdrantClient, models

from backend.core.errors import ProviderError
from backend.features.compliance.schemas import ComplianceResult, Evaluation
from backend.features.documents.chunking import ChunkPlan
from backend.integrations.embeddings.huggingface import HuggingFaceEmbeddings
from backend.integrations.llm.groq import GroqLLM
from backend.integrations.vector_store.qdrant import QdrantVectorStore, VectorRecord


def test_embedding_adapter_is_lazy_normalized_and_reused(settings):
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 3
    model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
    factory = Mock(return_value=model)
    embeddings = HuggingFaceEmbeddings(settings, factory)
    assert not embeddings.initialized
    assert embeddings.embed([]) == []
    factory.assert_not_called()
    assert embeddings.embed(["policy text"]) == [[1.0, 0.0, 0.0]]
    assert embeddings.dimension == 3
    embeddings.embed(["document text"])
    factory.assert_called_once()
    assert factory.call_args.kwargs["trust_remote_code"] is True
    assert factory.call_args.kwargs["local_files_only"] is True
    assert factory.call_args.kwargs["device"] == "cpu"
    assert model.encode.call_args.kwargs["normalize_embeddings"] is True
    embeddings.close()
    assert not embeddings.initialized


def test_embedding_errors(settings):
    embeddings = HuggingFaceEmbeddings(settings, Mock(side_effect=RuntimeError("download failed")))
    with pytest.raises(ValueError):
        embeddings.embed([""])
    with pytest.raises(ProviderError) as error:
        embeddings.embed(["text"])
    assert error.value.code == "embedding_initialization_failed"
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 3
    model.encode.return_value = np.array([[1.0, 0.0]])
    with pytest.raises(ProviderError) as error:
        HuggingFaceEmbeddings(settings, Mock(return_value=model)).embed(["text"])
    assert error.value.code == "embedding_dimension_invalid"


def test_qdrant_upsert_query_filter_and_persistence(settings):
    store = QdrantVectorStore(settings, lambda: 3)
    assert not store.initialized
    first, second = uuid4(), uuid4()
    try:
        store.upsert(
            [
                VectorRecord(
                    id=first,
                    vector=[1.0, 0.0, 0.0],
                    payload={"policy_id": "a", "text": "Privacy"},
                ),
                VectorRecord(
                    id=second,
                    vector=[0.0, 1.0, 0.0],
                    payload={"policy_id": "b", "text": "Retention"},
                ),
            ]
        )
        assert store.query([1, 0, 0], limit=1)[0].id == str(first)
        results = store.query(
            [1, 0, 0],
            query_filter=models.Filter(
                must=[models.FieldCondition(key="policy_id", match=models.MatchValue(value="b"))]
            ),
        )
        assert [hit.id for hit in results] == [str(second)]
        with pytest.raises(ValueError, match="dimension"):
            store.query([1, 0])
        with pytest.raises(ValueError, match="finite"):
            store.query([float("nan"), 0, 0])
        with pytest.raises(ValueError, match="nonzero"):
            store.query([0, 0, 0])
    finally:
        store.close()
    reopened = QdrantVectorStore(settings, lambda: 3)
    try:
        assert reopened.query([1, 0, 0], limit=1)[0].payload["text"] == "Privacy"
    finally:
        reopened.close()


def test_qdrant_rejects_existing_dimension_mismatch(settings):
    client = QdrantClient(path=str(settings.qdrant_path))
    client.create_collection(
        settings.qdrant_collection,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.close()
    store = QdrantVectorStore(settings, lambda: 3)
    try:
        with pytest.raises(ProviderError) as error:
            store.ensure_collection()
        assert error.value.code == "vector_collection_mismatch"
        assert not store.collection_ready
    finally:
        store.close()


def test_qdrant_server_mode_does_not_create_local_storage(settings):
    settings.qdrant_url = "http://localhost:6333"
    factory = Mock()
    client = factory.return_value
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=models.VectorParams(size=3, distance=models.Distance.COSINE)
            )
        )
    )
    store = QdrantVectorStore(settings, lambda: 3, factory)
    try:
        store.ensure_collection()
        assert factory.call_args.kwargs["url"] == settings.qdrant_url
        assert "path" not in factory.call_args.kwargs
        assert not settings.qdrant_path.exists()
    finally:
        store.close()


def response_payload(content, finish_reason="stop", refusal=None):
    return {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
    }


def llm_with_transport(settings, handler):
    settings.groq_api_key = SecretStr("test-key")

    def factory(**kwargs):
        return Groq(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    return GroqLLM(settings, factory)


def test_groq_sdk_structured_output_and_shutdown(settings):
    result = {"summary": "Review needed", "verdict": "needs_review", "findings": []}
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response_payload(json.dumps(result)))

    llm = llm_with_transport(settings, handler)
    assert not llm.initialized
    try:
        output = llm.generate_structured(
            "Assess provided evidence", "Example document", ComplianceResult
        )
        assert output.model_dump(mode="json", exclude_defaults=True) == result
        body = json.loads(requests[0].content)
        assert requests[0].url.host == "api.groq.com"
        assert requests[0].url.path == "/openai/v1/chat/completions"
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["messages"] == [
            {"role": "system", "content": "Assess provided evidence"},
            {"role": "user", "content": "Example document"},
        ]
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        schema = body["response_format"]["json_schema"]["schema"]
        assert set(schema["required"]) == set(schema["properties"])
        assert "default" not in schema["properties"]["comparison_matrix"]
        assert "store" not in body
    finally:
        client = llm._client
        llm.close()
    assert client.is_closed()
    assert not llm.initialized


def test_groq_missing_key_is_explicit(settings):
    factory = Mock()
    settings.groq_api_key = None
    with pytest.raises(ProviderError) as error:
        GroqLLM(settings, factory).generate_structured("Instructions", "Document", ComplianceResult)
    assert error.value.code == "llm_not_configured"
    assert "GROQ_API_KEY" in error.value.message
    factory.assert_not_called()


@pytest.mark.parametrize(
    "schema,value",
    [
        (ChunkPlan, {"groups": [{"unit_ids": [0], "topic": "Strength"}]}),
        (
            Evaluation,
            {
                "requirement": "Strength",
                "satisfaction": "insufficient_evidence",
                "extent_percent": None,
                "confidence": 0.5,
                "explanation": "Certificate missing",
                "policy_evidence": [],
                "supplier_evidence": [],
                "clarification_questions": ["Please provide a certificate."],
            },
        ),
    ],
)
def test_workflow_schemas_roundtrip_through_groq_sdk(settings, schema, value):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response_payload(json.dumps(value)))

    llm = llm_with_transport(settings, handler)
    try:
        assert (
            llm.generate_structured("Instructions", "Source evidence", schema).model_dump(
                mode="json"
            )
            == value
        )
        output_schema = requests[0]["response_format"]["json_schema"]["schema"]
        assert set(output_schema["required"]) == set(output_schema["properties"])
        assert output_schema["additionalProperties"] is False
    finally:
        llm.close()


@pytest.mark.parametrize(
    "failure,code",
    [
        ("refusal", "llm_refusal"),
        ("timeout", "llm_timeout"),
        ("auth", "llm_auth_failed"),
        ("rate_limit", "llm_rate_limited"),
        ("api", "llm_api_error"),
        ("invalid", "llm_invalid_output"),
        ("incomplete", "llm_incomplete"),
    ],
)
def test_groq_failure_modes(settings, failure, code):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if failure in {"api", "auth", "rate_limit"}:
            status = {"api": 500, "auth": 401, "rate_limit": 429}[failure]
            return httpx.Response(
                status,
                json={
                    "error": {
                        "message": "Upstream error",
                        "type": "invalid_request_error",
                        "code": "upstream_error",
                    }
                },
            )
        if failure == "refusal":
            return httpx.Response(200, json=response_payload(None, refusal="Declined"))
        if failure == "incomplete":
            return httpx.Response(200, json=response_payload(None, finish_reason="length"))
        return httpx.Response(200, json=response_payload("{}"))

    llm = llm_with_transport(settings, handler)
    try:
        with pytest.raises(ProviderError) as error:
            llm.generate_structured("Instructions", "Document", ComplianceResult)
        assert error.value.code == code
    finally:
        llm.close()
