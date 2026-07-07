"""Retrieval executor with per-stage candidate capture (PRD FR-4.2, FR-6.2, FR-7.2).

Replay mode (``adapter=None``) serves dense candidates from exact cosine over
the snapshot's stored embeddings, no serving index required (FR-4.2, FR-7.2).
Live mode queries the adapter's collection (named by ``ingest.collection_name``).
Replay normalizes rows and query exactly like ``LocalIndexAdapter`` and shares
its ``(-score, chunk_id)`` tie-break, so replay and an exact local index return
identical rankings.

Chunk resolution: manifests carry the store-relative parquet URI
(``artifacts/chunks/<chunkset_key>.parquet``). Chunkset keys are hex digests,
which the store's filename sanitizer passes through unchanged, so the parquet
stem IS the chunkset key and ``store.get_chunks(stem)`` resolves it without
scanning the store.

``exact_dense_ranks`` is the FR-6.2 shadow scorer: full-corpus exact cosine
from stored embeddings regardless of the adapter, so funnel attribution can
compare a live (possibly approximate) dense ranking against ground truth.

Candidate depth: dense and sparse each retrieve ``max(top_k * 4, 20)``
candidates before fusion; the final list truncates to ``top_k`` (or the rerank
stage's ``top_n``). When a rerank stage is present it reorders the fused top
``top_k`` candidates and its output becomes ``final``. Sparse candidates are
computed even without a hybrid block so funnel attribution always has every
stage; ``fused`` is dense-order in that case.
"""
from __future__ import annotations

from pathlib import PurePosixPath

import numpy as np

from . import hashing
from .adapters.base import VectorAdapter
from .bm25 import BM25Index
from .fusion import fuse
from .ingest import collection_name, embedding_keys
from .models import ChunkRecord, QueryRun, SnapshotManifest, StageCandidates
from .pipeline.providers import EmbeddingProvider, get_provider
from .rerankers import get_reranker
from .store import ProjectStore

__all__ = ["RetrievalEngine", "collection_name", "chunkset_key_from_uri",
           "candidate_depth", "MIN_CANDIDATE_DEPTH"]

MIN_CANDIDATE_DEPTH = 20


def chunkset_key_from_uri(chunks_uri: str) -> str:
    return PurePosixPath(chunks_uri).stem


def candidate_depth(top_k: int) -> int:
    return max(top_k * 4, MIN_CANDIDATE_DEPTH)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


class RetrievalEngine:
    def __init__(self, store: ProjectStore, manifest: SnapshotManifest,
                 adapter: VectorAdapter | None = None) -> None:
        self.store = store
        self.manifest = manifest
        self.adapter = adapter
        self._chunks: list[ChunkRecord] | None = None
        self._provider: EmbeddingProvider | None = None
        self._bm25: BM25Index | None = None
        self._dense_index: tuple[list[str], np.ndarray] | None = None
        self._doc_map: dict[str, str] | None = None

    def chunks(self) -> list[ChunkRecord]:
        if self._chunks is None:
            key = chunkset_key_from_uri(self.manifest.artifacts["chunks_uri"])
            self._chunks = self.store.get_chunks(key)
        return self._chunks

    def chunk_texts(self) -> dict[str, str]:
        return {r.chunk_id: r.text for r in self.chunks()}

    def chunk_doc_map(self) -> dict[str, str]:
        if self._doc_map is None:
            docs = self.store.docs_for_merkle(self.manifest.corpus.merkle_root)
            by_doc = {d["doc_id"]: d["source_path"] for d in docs}
            self._doc_map = {
                r.chunk_id: by_doc[r.doc_id] if r.doc_id in by_doc
                else self.store.get_doc(r.doc_id)["source_path"]
                for r in self.chunks()
            }
        return self._doc_map

    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            embed = self.manifest.pipeline.stage("embed")
            if embed is None:
                raise ValueError("manifest pipeline has no 'embed' stage")
            self._provider = get_provider(dict(embed.params))
        return self._provider

    def query_vector(self, question: str) -> np.ndarray:
        p = self.provider
        key = hashing.embedding_key(hashing.text_hash(question), p.provider, p.model,
                                    p.dims, hashing.params_hash(p.params))
        cached = self.store.query_vector_cached(key)
        if cached is not None:
            return cached
        vec = p.embed([question])[0]
        self.store.cache_query_vector(key, vec)
        return vec

    def exact_dense_ranks(self, question: str) -> list[tuple[str, float]]:
        chunk_ids, matrix = self._dense_matrix()
        if not chunk_ids:
            return []
        query = _l2_normalize(np.asarray(self.query_vector(question), dtype=np.float32).reshape(-1))
        cosine = matrix @ query
        order = sorted(range(len(chunk_ids)), key=lambda i: (-cosine[i], chunk_ids[i]))
        return [(chunk_ids[i], float(cosine[i])) for i in order]

    def run_query(self, query_id: str, question: str) -> QueryRun:
        retrieve = self.manifest.pipeline.stage("retrieve")
        params = retrieve.params if retrieve is not None else {}
        top_k = int(params.get("top_k", 10))
        depth = candidate_depth(top_k)

        dense = self._dense_candidates(question, depth)
        sparse = self._bm25_index().top(question, depth)
        hybrid = params.get("hybrid")
        if hybrid:
            fused = fuse(dense, sparse, str(hybrid.get("fusion", "weighted")), dict(hybrid))
        else:
            fused = list(dense)

        rerank_stage = self.manifest.pipeline.stage("rerank")
        if rerank_stage is not None:
            reranker = get_reranker(rerank_stage.tool, dict(rerank_stage.params))
            texts = self.chunk_texts()
            candidates = [(cid, texts[cid]) for cid, _ in fused[:top_k]]
            reranked = reranker(question, candidates)
            top_n = int(rerank_stage.params.get("top_n", top_k))
            final = list(reranked[:top_n])
        else:
            reranked = None
            final = list(fused[:top_k])

        return QueryRun(
            query_id=query_id,
            question=question,
            stages=StageCandidates(dense=list(dense), sparse=list(sparse),
                                   fused=list(fused), reranked=reranked),
            final=final,
        )

    def _dense_candidates(self, question: str, depth: int) -> list[tuple[str, float]]:
        if self.adapter is None:
            return self.exact_dense_ranks(question)[:depth]
        return self.adapter.query_dense(
            collection_name(self.manifest, self.store.project),
            self.query_vector(question), depth)

    def _dense_matrix(self) -> tuple[list[str], np.ndarray]:
        if self._dense_index is None:
            records = self.chunks()
            keys = embedding_keys(records, self.provider)
            found = self.store.get_embeddings(keys)
            missing = [k for k in keys if k not in found]
            if missing:
                raise KeyError(
                    f"{len(missing)} embeddings missing from store for snapshot "
                    f"{self.manifest.snapshot_id} (first: {missing[0]})"
                )
            if records:
                matrix = _l2_normalize(np.stack([found[k] for k in keys]).astype(np.float32))
            else:
                matrix = np.zeros((0, self.provider.dims), dtype=np.float32)
            self._dense_index = ([r.chunk_id for r in records], matrix)
        return self._dense_index

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.chunk_texts())
        return self._bm25
