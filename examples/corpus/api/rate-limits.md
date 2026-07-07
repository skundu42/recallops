# API Rate Limits

## Default quotas

The default quota is 600 requests per minute per project for read endpoints and 120
requests per minute for write endpoints. Batch endpoints count each item in the
batch as one write request. Quotas are enforced with a sliding window, not fixed
buckets, so short bursts above the steady rate are tolerated.

## Burst and backoff

When a request is throttled, the response is HTTP 429 with a Retry-After header in
seconds. Clients should implement exponential backoff with jitter, starting at one
second and capping at 60 seconds. Retrying without backoff extends the throttle
window.

## Raising limits

Team and Enterprise plans can request higher quotas from the console's usage page.
Increases up to 5x the default are approved automatically; larger increases require
a capacity review, which typically completes in 2 business days.

## Streaming endpoints

Streaming endpoints are limited by concurrent connections rather than request rate:
20 concurrent streams per project by default. Opening a 21st stream closes the
oldest idle stream with code STREAM_EVICTED.

## Rate limit headers

Every API response includes X-RateLimit-Limit, X-RateLimit-Remaining, and
X-RateLimit-Reset headers. Dashboards should read these headers rather than
counting requests client-side, because server-side quotas account for batch
expansion and internal retries.
