from __future__ import annotations

import numpy as np
import pytest

import recallops.pipeline.providers as providers_module
from recallops import hashing
from recallops.pipeline.providers import (
    CohereProvider,
    EmbeddingProvider,
    LocalHashProvider,
    OpenAIProvider,
    VoyageProvider,
    cohere_request_body,
    embed_stage_spec,
    estimate_embed_cost,
    get_provider,
    parse_cohere_embeddings,
    parse_voyage_embeddings,
    voyage_request_body,
)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class TestLocalHashProvider:
    def test_determinism_across_instances(self):
        texts = ["the refund policy covers annual plans", "gateway timeouts default to thirty seconds"]
        a = LocalHashProvider().embed(texts)
        b = LocalHashProvider().embed(texts)
        assert np.array_equal(a, b)

    def test_shape_dtype_and_unit_norm(self):
        vecs = LocalHashProvider(dims=128).embed(["alpha beta gamma", "delta epsilon"])
        assert vecs.shape == (2, 128)
        assert vecs.dtype == np.float32
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)

    def test_pure_function_of_text(self):
        p = LocalHashProvider()
        both = p.embed(["first text here", "second text there"])
        solo_first = p.embed(["first text here"])
        solo_second = p.embed(["second text there"])
        assert np.array_equal(both[0], solo_first[0])
        assert np.array_equal(both[1], solo_second[0])

    def test_seed_alters_mapping(self):
        text = ["the same identical sentence for both seeds"]
        v0 = LocalHashProvider(seed=0).embed(text)[0]
        v1 = LocalHashProvider(seed=7).embed(text)[0]
        assert not np.allclose(v0, v1)

    def test_model_alters_mapping(self):
        text = ["the same identical sentence for both models"]
        v0 = LocalHashProvider(model="hash-v1").embed(text)[0]
        v1 = LocalHashProvider(model="hash-v1b").embed(text)[0]
        assert not np.allclose(v0, v1)

    def test_ngram_range_alters_vectors(self):
        text = ["feature hashing uses word bigrams too"]
        uni = LocalHashProvider(ngram=(1, 1)).embed(text)[0]
        unibi = LocalHashProvider(ngram=(1, 2)).embed(text)[0]
        assert not np.allclose(uni, unibi)

    def test_empty_text_is_finite_zero_vector(self):
        vecs = LocalHashProvider().embed(["", "   "])
        assert np.all(np.isfinite(vecs))
        assert np.allclose(vecs, 0.0)

    def test_similar_texts_closer_than_unrelated_on_corpus(self, corpus_dir):
        refunds = (corpus_dir / "billing" / "refunds.md").read_text()
        invoices = (corpus_dir / "billing" / "invoices.md").read_text()
        architecture = (corpus_dir / "eng" / "architecture.md").read_text()
        half = len(refunds) // 2
        p = LocalHashProvider()
        v_a, v_b, v_inv, v_arch = p.embed([refunds[:half], refunds[half:], invoices, architecture])
        assert _cos(v_a, v_b) > _cos(v_a, v_arch)
        assert _cos(v_a, v_inv) > _cos(v_a, v_arch)

    def test_price_is_zero(self):
        assert LocalHashProvider().price_per_1k_tokens() == 0.0

    def test_default_params(self):
        p = LocalHashProvider()
        assert p.provider == "local"
        assert p.model == "hash-v1"
        assert p.dims == 256
        assert p.params == {"dims": 256, "seed": 0, "ngram": [1, 2]}

    def test_model_key_stability(self):
        p1 = LocalHashProvider()
        p2 = LocalHashProvider()
        expected = f"local_hash-v1_256_{hashing.params_hash(p1.params)}"
        assert p1.model_key == expected
        assert p1.model_key == p2.model_key
        assert LocalHashProvider(seed=7).model_key != p1.model_key
        assert LocalHashProvider(dims=128).model_key != p1.model_key


class TestOpenAIProvider:
    def test_price_table(self):
        assert OpenAIProvider("text-embedding-3-small").price_per_1k_tokens() == pytest.approx(0.00002)
        assert OpenAIProvider("text-embedding-3-large").price_per_1k_tokens() == pytest.approx(0.00013)

    def test_unknown_model_rejected(self):
        with pytest.raises(ValueError):
            OpenAIProvider("text-embedding-9-huge")

    def test_default_dims_per_model(self):
        assert OpenAIProvider("text-embedding-3-small").dims == 1536
        assert OpenAIProvider("text-embedding-3-large").dims == 3072
        assert OpenAIProvider("text-embedding-3-small", dims=256).dims == 256

    def test_model_key(self):
        p = OpenAIProvider("text-embedding-3-small", dims=512)
        assert p.model_key == f"openai_text-embedding-3-small_512_{hashing.params_hash(p.params)}"

    def test_embed_without_key_raises_before_network(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            OpenAIProvider("text-embedding-3-small").embed(["hello"])

    def test_is_embedding_provider(self):
        assert isinstance(OpenAIProvider("text-embedding-3-small"), EmbeddingProvider)


class TestStageSpecAndFactory:
    def test_embed_stage_spec_shape(self):
        spec = embed_stage_spec("local", "hash-v1", 256, {"seed": 0, "ngram": (1, 2)})
        assert spec.id == "embed"
        assert spec.params["provider"] == "local"
        assert spec.params["model"] == "hash-v1"
        assert spec.params["dims"] == 256
        hashing.canonical_json(spec.params)
        assert spec.params_hash == embed_stage_spec("local", "hash-v1", 256, {"seed": 0, "ngram": (1, 2)}).params_hash

    def test_get_provider_local_round_trip(self):
        spec = embed_stage_spec("local", "hash-v1", 128, {"seed": 3, "ngram": (1, 2)})
        p = get_provider(spec.params)
        assert isinstance(p, LocalHashProvider)
        assert p.dims == 128
        assert p.params["seed"] == 3
        assert p.model_key == get_provider(spec.params).model_key
        direct = LocalHashProvider(dims=128, seed=3)
        assert np.array_equal(p.embed(["same text"]), direct.embed(["same text"]))

    def test_get_provider_defaults_for_local(self):
        p = get_provider({"provider": "local", "model": "hash-v1", "dims": 256})
        assert isinstance(p, LocalHashProvider)
        assert p.params == {"dims": 256, "seed": 0, "ngram": [1, 2]}

    def test_get_provider_openai(self):
        p = get_provider({"provider": "openai", "model": "text-embedding-3-large", "dims": 1024})
        assert isinstance(p, OpenAIProvider)
        assert p.dims == 1024

    def test_get_provider_unknown_raises(self):
        with pytest.raises(ValueError):
            get_provider({"provider": "nonexistent", "model": "m", "dims": 512})


class TestEstimateEmbedCost:
    def test_token_and_usd_math(self):
        texts = ["a" * 40, "b" * 9]
        est = estimate_embed_cost(OpenAIProvider("text-embedding-3-small"), texts)
        assert est["n_texts"] == 2
        assert est["est_tokens"] == 12
        assert est["usd"] == pytest.approx(12 / 1000 * 0.00002)
        assert est["wall_s"] >= 0.0

    def test_local_is_free(self):
        est = estimate_embed_cost(LocalHashProvider(), ["hello world" * 20])
        assert est["usd"] == 0.0
        assert est["est_tokens"] == len("hello world" * 20) // 4
        assert est["n_texts"] == 1

    def test_empty_texts(self):
        est = estimate_embed_cost(LocalHashProvider(), [])
        assert est == {"n_texts": 0, "est_tokens": 0, "usd": 0.0, "wall_s": 0.0}

    def test_deterministic(self):
        texts = ["x" * 100, "y" * 55]
        p = OpenAIProvider("text-embedding-3-large")
        assert estimate_embed_cost(p, texts) == estimate_embed_cost(p, texts)


class TestEmbedQueries:
    def test_default_embed_queries_equals_embed(self):
        p = LocalHashProvider()
        texts = ["what is the refund window?", "how are invoices numbered?"]
        assert np.array_equal(p.embed_queries(texts), p.embed(texts))

    def test_subclass_override_is_used(self):
        class Asymmetric(LocalHashProvider):
            def embed_queries(self, texts):
                return -super().embed(texts)

        p = Asymmetric()
        assert np.array_equal(p.embed_queries(["q"]), -p.embed(["q"]))


class TestCohereProvider:
    def test_request_body_documents(self):
        body = cohere_request_body("embed-english-v3.0", ["a", "b"], "search_document")
        assert body == {"model": "embed-english-v3.0", "texts": ["a", "b"],
                        "input_type": "search_document", "embedding_types": ["float"]}

    def test_parse_embeddings(self):
        payload = {"embeddings": {"float": [[1.0, 0.0], [0.0, 1.0]]}}
        assert parse_cohere_embeddings(payload) == [[1.0, 0.0], [0.0, 1.0]]

    def test_unknown_model_rejected(self):
        with pytest.raises(ValueError):
            CohereProvider("embed-english-v99")

    def test_dims_must_match_model(self):
        with pytest.raises(ValueError):
            CohereProvider("embed-english-v3.0", dims=512)
        assert CohereProvider("embed-english-v3.0", dims=1024).dims == 1024

    def test_embed_without_key_raises_before_network(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            CohereProvider().embed(["hello"])

    def test_embed_uses_document_input_type_and_normalizes(self, monkeypatch):
        seen = []

        def fake_post(url, body, headers):
            seen.append((url, body, headers))
            return {"embeddings": {"float": [[3.0, 4.0] + [0.0] * 1022
                                             for _ in body["texts"]]}}

        monkeypatch.setattr(providers_module, "_post_json", fake_post)
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        p = CohereProvider()
        mat = p.embed(["x", "y"])
        assert mat.shape == (2, 1024)
        assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)
        assert seen[0][1]["input_type"] == "search_document"
        assert seen[0][2]["Authorization"] == "Bearer test-key"

    def test_embed_queries_uses_query_input_type(self, monkeypatch):
        seen = []

        def fake_post(url, body, headers):
            seen.append(body)
            return {"embeddings": {"float": [[1.0] + [0.0] * 1023
                                             for _ in body["texts"]]}}

        monkeypatch.setattr(providers_module, "_post_json", fake_post)
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        CohereProvider().embed_queries(["what is x?"])
        assert seen[0]["input_type"] == "search_query"

    def test_price_positive_so_cost_gate_engages(self):
        assert CohereProvider().price_per_1k_tokens() > 0.0

    def test_model_key_shape(self):
        p = CohereProvider()
        assert p.model_key == (
            f"cohere_embed-english-v3.0_1024_{hashing.params_hash(p.params)}"
        )


class TestVoyageProvider:
    def test_request_body(self):
        body = voyage_request_body("voyage-3.5", ["a"], "document")
        assert body == {"model": "voyage-3.5", "input": ["a"], "input_type": "document"}

    def test_parse_embeddings_sorts_by_index(self):
        payload = {"data": [{"index": 1, "embedding": [0.0, 1.0]},
                            {"index": 0, "embedding": [1.0, 0.0]}]}
        assert parse_voyage_embeddings(payload) == [[1.0, 0.0], [0.0, 1.0]]

    def test_unknown_model_rejected(self):
        with pytest.raises(ValueError):
            VoyageProvider("voyage-99")

    def test_embed_without_key_raises_before_network(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            VoyageProvider().embed(["hello"])

    def test_embed_queries_uses_query_input_type(self, monkeypatch):
        seen = []

        def fake_post(url, body, headers):
            seen.append(body)
            return {"data": [{"index": i, "embedding": [1.0] + [0.0] * 1023}
                             for i in range(len(body["input"]))]}

        monkeypatch.setattr(providers_module, "_post_json", fake_post)
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        p = VoyageProvider()
        p.embed_queries(["q"])
        p.embed(["d"])
        assert seen[0]["input_type"] == "query"
        assert seen[1]["input_type"] == "document"

    def test_embed_normalizes_and_authorizes(self, monkeypatch):
        seen = []

        def fake_post(url, body, headers):
            seen.append((url, body, headers))
            return {"data": [{"index": i, "embedding": [3.0, 4.0] + [0.0] * 1022}
                             for i in range(len(body["input"]))]}

        monkeypatch.setattr(providers_module, "_post_json", fake_post)
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        p = VoyageProvider()
        mat = p.embed(["x", "y"])
        assert mat.shape == (2, 1024)
        assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)
        assert seen[0][1]["input_type"] == "document"
        assert seen[0][2]["Authorization"] == "Bearer test-key"

    def test_model_key_shape(self):
        p = VoyageProvider()
        assert p.model_key == f"voyage_voyage-3.5_1024_{hashing.params_hash(p.params)}"

    def test_price_positive_so_cost_gate_engages(self):
        assert VoyageProvider().price_per_1k_tokens() > 0.0


class TestGetProviderNewBackends:
    def test_cohere_dispatch(self):
        p = get_provider({"provider": "cohere", "model": "embed-english-v3.0", "dims": 1024})
        assert isinstance(p, CohereProvider)

    def test_voyage_dispatch(self):
        p = get_provider({"provider": "voyage", "model": "voyage-3.5", "dims": 1024})
        assert isinstance(p, VoyageProvider)
