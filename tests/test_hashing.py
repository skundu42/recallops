from __future__ import annotations

from recallops import hashing


def test_canonical_json_ignores_key_order():
    assert hashing.canonical_json({"b": 1, "a": [2, {"y": 0, "x": 1}]}) == \
        hashing.canonical_json({"a": [2, {"x": 1, "y": 0}], "b": 1})


def test_h_is_stable_and_part_sensitive():
    assert hashing.h("a", "b") == hashing.h("a", "b")
    assert hashing.h("a", "b") != hashing.h("ab")
    assert hashing.h("a", "b") != hashing.h("b", "a")
    assert len(hashing.h("x")) == 16


def test_id_prefixes():
    assert hashing.doc_id(b"bytes").startswith("doc_")
    assert hashing.text_hash("t").startswith("tx_")
    assert hashing.chunk_id("doc_1", 0, 5, "hello").startswith("ch_")
    assert hashing.params_hash({"k": 1}).startswith("ph_")
    assert hashing.embedding_key("tx_1", "local", "hash-v1", 256, "ph_1").startswith("emb_")
    assert hashing.merkle_root([("a.md", "doc_1")]).startswith("mr_")
    assert hashing.snapshot_hash({"x": 1}).startswith("snap_")


def test_chunk_id_span_sensitive():
    a = hashing.chunk_id("doc_1", 0, 5, "hello")
    assert a != hashing.chunk_id("doc_1", 1, 6, "hello")
    assert a != hashing.chunk_id("doc_2", 0, 5, "hello")
    assert a == hashing.chunk_id("doc_1", 0, 5, "hello")


def test_merkle_root_order_invariant():
    pairs = [("b.md", "doc_2"), ("a.md", "doc_1")]
    assert hashing.merkle_root(pairs) == hashing.merkle_root(list(reversed(pairs)))
    assert hashing.merkle_root(pairs) != hashing.merkle_root([("a.md", "doc_1")])


def test_params_hash_matches_canonical_semantics():
    assert hashing.params_hash({"a": 1, "b": 2}) == hashing.params_hash({"b": 2, "a": 1})
    assert hashing.params_hash({"a": 1}) != hashing.params_hash({"a": 2})
