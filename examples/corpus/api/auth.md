# API Authentication

## API keys

Every request to the REST API must include an API key in the Authorization header
using the Bearer scheme. Keys are scoped per project and carry one of three roles:
read, write, or admin. Keys are shown once at creation time and stored only as a
salted hash; a lost key cannot be recovered, only rotated.

## Key rotation

Rotating a key creates a replacement immediately while the old key continues to
work for a configurable overlap window, 24 hours by default. The overlap window
exists so deployed services can pick up the new secret without an outage. Rotation
events are recorded in the audit log with the actor and source IP.

## OAuth service accounts

Machine-to-machine integrations can use OAuth 2.0 client credentials instead of
static keys. Tokens issued this way expire after 60 minutes and must be refreshed;
refresh does not require re-authentication. Client secrets can be rotated with the
same 24-hour overlap semantics as API keys.

## IP allowlists

Projects can restrict API access to an IP allowlist in CIDR notation. Requests from
outside the allowlist receive HTTP 403 with error code IP_NOT_ALLOWED. The allowlist
applies to API keys but not to the browser console, which uses session auth.

## Common errors

401 UNAUTHENTICATED means the key is missing, malformed, or revoked. 403 PERMISSION_DENIED
means the key's role does not permit the operation. 429 indicates rate limiting; see
the rate limits page for quotas and backoff guidance.
