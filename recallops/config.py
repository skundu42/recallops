"""Project configuration (`recall.yaml`) and adapter/provider construction.

``recall.yaml`` is authored and re-read through a tiny YAML subset
(``parse_simple_yaml`` / ``dump_simple_yaml``) so RecallOps carries no
third-party YAML dependency. The subset is exactly what a project config needs:
nested maps by two-space indentation, scalars (str/int/float/bool/null) and
inline lists (``[a, b]``). Round-trips are lossless for that subset.

``ProjectConfig.pipeline`` is stored in the shape ``ingest.build_pipeline``
consumes, so the config *is* the pipeline definition. ``build_adapter`` and
``build_provider`` turn a config into the runtime objects the CLI wires
together.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

from .adapters.base import VectorAdapter
from .adapters.local import LocalIndexAdapter
from .ingest import DEFAULT_CONFIG
from .pipeline.providers import EmbeddingProvider, get_provider
from .store import ProjectStore

__all__ = [
    "ProjectConfig",
    "parse_simple_yaml",
    "dump_simple_yaml",
    "build_adapter",
    "build_provider",
    "provider_from_embedding",
    "parse_embedding_spec",
]

_DEFAULT_GATE = {"mode": "statistical", "primary_metric": "recall@5", "q": 0.05}
_DEFAULT_COSTS = {"max_cost": 0.0}


def _parse_scalar(value: str):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in inner.split(",")]
    if (len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'"):
        return value[1:-1]
    low = value.lower()
    if low in ("null", "~", "none", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_simple_yaml(text: str) -> dict:
    """Parse the RecallOps YAML subset into a nested dict.

    Supports nested maps by two-space indentation, ``key: value`` scalars
    (str/int/float/bool/null), and inline lists ``[a, b]``. Blank lines and
    full-line ``#`` comments are ignored.
    """
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        content = raw.strip()
        key, sep, value = content.partition(":")
        if not sep:
            raise ValueError(f"invalid config line (missing ':'): {raw!r}")
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


_QUOTE_TRIGGERS = set("[],:#{}")


def _dump_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_dump_scalar(v) for v in value) + "]"
    text = str(value)
    needs_quote = (
        text == ""
        or text != text.strip()
        or text.lower() in ("null", "true", "false", "~", "none")
        or any(ch in _QUOTE_TRIGGERS for ch in text)
    )
    if not needs_quote:
        try:
            float(text)
            needs_quote = True
        except ValueError:
            needs_quote = False
    return f'"{text}"' if needs_quote else text


def _dump_map(data: dict, indent: int, lines: list[str]) -> None:
    pad = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            _dump_map(value, indent + 1, lines)
        else:
            lines.append(f"{pad}{key}: {_dump_scalar(value)}")


def dump_simple_yaml(data: dict) -> str:
    """Serialize a nested dict to the RecallOps YAML subset (round-trips with
    ``parse_simple_yaml``)."""
    lines: list[str] = []
    _dump_map(data, 0, lines)
    return "\n".join(lines) + "\n"


@dataclass
class ProjectConfig:
    project: str
    source: str
    adapter: dict
    pipeline: dict
    gate: dict
    costs: dict

    @classmethod
    def default(cls, project: str, source: str = "docs", adapter: str = "local",
                dsn: str | None = None) -> ProjectConfig:
        adapter_block: dict = {"type": adapter}
        if adapter == "pgvector":
            adapter_block["dsn"] = dsn or ""
        return cls(
            project=project,
            source=source,
            adapter=adapter_block,
            pipeline=copy.deepcopy(DEFAULT_CONFIG),
            gate=dict(_DEFAULT_GATE),
            costs=dict(_DEFAULT_COSTS),
        )

    def to_dict(self) -> dict:
        return {
            "project": self.project,
            "source": self.source,
            "adapter": self.adapter,
            "pipeline": self.pipeline,
            "gate": self.gate,
            "costs": self.costs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProjectConfig:
        return cls(
            project=d.get("project", "recall-project"),
            source=d.get("source", "docs"),
            adapter=d.get("adapter", {"type": "local"}),
            pipeline=d.get("pipeline", copy.deepcopy(DEFAULT_CONFIG)),
            gate={**_DEFAULT_GATE, **d.get("gate", {})},
            costs={**_DEFAULT_COSTS, **d.get("costs", {})},
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(dump_simple_yaml(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ProjectConfig:
        return cls.from_dict(parse_simple_yaml(Path(path).read_text(encoding="utf-8")))


def provider_from_embedding(embed: dict) -> EmbeddingProvider:
    spec = {
        "provider": embed.get("provider", "local"),
        "model": embed["model"],
        "dims": int(embed["dims"]),
    }
    spec.update(embed.get("params", {}))
    return get_provider(spec)


def build_provider(cfg: ProjectConfig) -> EmbeddingProvider:
    return provider_from_embedding(cfg.pipeline["embedding"])


def build_adapter(cfg: ProjectConfig, store: ProjectStore) -> VectorAdapter:
    block = cfg.adapter or {}
    kind = block.get("type", "local")
    if kind == "local":
        return LocalIndexAdapter(store.base)
    if kind == "pgvector":
        from .adapters.pgvector import PgVectorAdapter

        dsn = block.get("dsn") or os.environ.get("RECALL_PG_DSN")
        if not dsn:
            raise ValueError("pgvector adapter requires a dsn (config adapter.dsn or RECALL_PG_DSN)")
        kwargs = {}
        if block.get("schema"):
            kwargs["schema"] = block["schema"]
        if block.get("probes") is not None:
            kwargs["probes"] = int(block["probes"])
        return PgVectorAdapter(dsn, **kwargs)
    raise ValueError(f"unknown adapter type {kind!r}; expected 'local' or 'pgvector'")


_LOCAL_DEFAULT_PARAMS = {"seed": 0, "ngram": [1, 2]}
_DEFAULT_SPEC_DIMS = 256


def parse_embedding_spec(spec: str) -> dict:
    """Parse ``provider:model:dims`` (or a bare local model name) into an
    embedding-section dict for ``build_pipeline``."""
    parts = spec.split(":")
    if len(parts) >= 3:
        provider, model, dims = parts[0], parts[1], int(parts[2])
    elif len(parts) == 2:
        provider, model, dims = parts[0], parts[1], _DEFAULT_SPEC_DIMS
    else:
        provider, model, dims = "local", spec, _DEFAULT_SPEC_DIMS
    params = dict(_LOCAL_DEFAULT_PARAMS) if provider == "local" else {}
    return {"provider": provider, "model": model, "dims": dims, "params": params}
