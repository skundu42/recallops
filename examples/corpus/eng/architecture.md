# System Architecture

## Service topology

The platform is a modular monolith fronted by an edge proxy, with three satellite
services: the ingestion worker pool, the query gateway, and the async job runner.
Services communicate over gRPC internally; the public surface is REST plus a
streaming websocket endpoint.

## Data stores

Primary state lives in Postgres with logical replication to a warm standby.
Vector indexes are served from a dedicated cluster; object storage holds raw
documents and build artifacts. Redis is used only for ephemeral queues and rate
counters, nothing in Redis is a source of truth.

## Deployment model

Deploys are blue-green with automatic rollback on SLO burn. Database migrations
follow expand-migrate-contract: additive schema change first, dual-write window,
then contraction after a full retention cycle. Feature flags gate all user-visible
changes.

## Multi-tenancy

Tenancy is enforced at the row level with per-tenant encryption keys for content
columns. Noisy-neighbor protection combines per-tenant rate limits at the gateway
with weighted fair queuing in the job runner.

## Observability

Every request carries a trace ID from edge to storage. Golden signals (latency,
traffic, errors, saturation) alert on burn rate against SLOs, not static
thresholds. Logs are structured, sampled at the edge, and retained for 30 days.
