# Incident Runbook

## Declaring an incident

Any engineer can declare an incident by running the incident bot command in the
operations channel. Declaring creates a dedicated channel, pages the incident
commander rotation, and starts the timeline recorder. When in doubt, declare; a
false alarm costs minutes, an undeclared outage costs trust.

## Roles

The incident commander owns coordination and communications and never debugs. The
operations lead drives technical mitigation. The scribe keeps the timeline current.
Handoffs are announced in the incident channel with an explicit "you have command"
acknowledgement.

## Mitigation before diagnosis

Prefer reversible mitigations first: roll back the last deploy, shift traffic to
the healthy region, enable the static fallback. Root-cause analysis happens after
customer impact stops, not before. Every mitigation is timestamped in the timeline.

## Communication cadence

Status page updates go out within 15 minutes of declaration and at least every 30
minutes thereafter until resolution. Internal stakeholders get a summary at
declaration, at each material change, and at resolution.

## Postmortems

Every severity 1 or 2 incident gets a blameless postmortem within 5 business days,
with contributing factors, what went well, and tracked action items. Action items
have owners and due dates and are reviewed weekly until closed.
