# Security Policy

## Supported versions

RecallOps is pre-1.0 and under active development. Security fixes are applied to
the latest released minor series and to `main`. Older pre-1.0 versions do not
receive backported fixes; please upgrade to the latest release.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

Once RecallOps reaches 1.0, this table will be updated to state the concrete
support window for each release line.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report privately, in order of preference:

1. **GitHub private advisory**: use the repository's
   **Security → Report a vulnerability** page
   (`https://github.com/OWNER/recallops/security/advisories/new`), which opens a
   private [GitHub Security Advisory](https://docs.github.com/en/code-security/security-advisories).
2. **Email**: `security@OWNER` _(placeholder: replace `OWNER` with the
   project's real security contact address before publishing)_.

Please include, as far as you can:

- a description of the issue and its impact,
- the affected version(s) and platform (OS, Python version, adapter),
- a minimal reproduction or proof of concept,
- and any suggested remediation.

## Disclosure expectations

- We will **acknowledge** your report within **3 business days**.
- We aim to provide an initial **assessment within 7 business days**, and to
  ship a fix or a documented mitigation as quickly as the severity warrants.
- We practice **coordinated disclosure**: please give us a reasonable window
  (target **90 days**, sooner for actively exploited issues) to release a fix
  before any public disclosure, and we will credit you in the advisory and
  release notes unless you prefer to remain anonymous.
- Please make a good-faith effort to avoid privacy violations, data destruction,
  and service disruption while researching. Testing should only ever be done
  against data and infrastructure you own or are explicitly authorized to use.

## Data handling and trust boundaries

RecallOps is designed to keep sensitive material inside customer-controlled
boundaries, which narrows its security surface:

- **Runs offline by default.** The default stack (local hash embeddings + the
  built-in exact-KNN adapter) makes **no network calls** and requires no API
  keys. In local mode, customer data (documents, chunks, and embeddings)
  never leaves customer-controlled storage.
- **LLM/embedding features use user-supplied keys only.** OpenAI embeddings,
  pgvector, and any other provider integration are strictly opt-in and run
  solely with credentials **you** supply (e.g. `OPENAI_API_KEY`, `RECALL_PG_DSN`).
  RecallOps does not ship, proxy, or phone-home any credentials, and any
  provider-billed operation is cost-gated and requires explicit approval.
- **Telemetry is opt-in and metadata-only.** RecallOps does not collect
  telemetry by default. If telemetry is ever enabled, it is limited to
  non-sensitive operational metadata (e.g. command name, version, timing) and
  **never** includes document text, chunk content, embeddings, queries, or
  credentials.

If you believe any of these boundaries can be crossed, for example a code path
that transmits corpus content or credentials off-box, or that bills a provider
without the cost gate, treat it as a security issue and report it through the
private channels above.
