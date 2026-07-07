# Single Sign-On

## Supported protocols

We support SAML 2.0 and OIDC for single sign-on. SAML assertions must be signed
with RSA-SHA256; unsigned assertions are rejected. OIDC integrations use the
authorization code flow with PKCE. Legacy protocols such as CAS or WS-Federation
are not supported.

## SCIM provisioning

User provisioning and deprovisioning are handled through SCIM 2.0. When an identity
provider deactivates a user, the SCIM sync revokes all active sessions within 15
minutes. Group membership pushed via SCIM maps directly to workspace roles.

## Enforcing SSO

Workspace admins can enforce SSO-only login, which disables password authentication
for all members except designated break-glass accounts. Break-glass accounts require
hardware security keys and are audited on every use.

## Session policy

SSO sessions default to an 8-hour lifetime and can be configured between 1 hour and
30 days. Revoking a user in the identity provider terminates the application
session at the next token refresh, at most 15 minutes later.

## Certificate rotation

Identity provider signing certificates can be rotated without downtime by uploading
the new certificate alongside the old one; both are accepted during a 72-hour grace
window. Expired certificates cause login failures with error code SSO_CERT_EXPIRED.
