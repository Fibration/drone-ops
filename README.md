# drone-ops
Agentic Multi-UAV Mission Control &amp; Incident Recovery


START
  ↓
FLEET MONITOR
  ↓
COMMANDER
  ├── telemetry needed ─────→ FLEET MONITOR
  │                              ↑
  ├── planning needed ──────→ PLANNER
  │                              │
  │                              ↓
  │                           COMMANDER
  │
  ├── plan ready ───────────→ SAFETY REVIEW
  │                              │
  │                 ┌────────────┼─────────────┐
  │                 ↓            ↓             ↓
  │             APPROVE       REVISE       HUMAN
  │                 │            │          APPROVAL
  │                 ↓            └→ PLANNER     │
  │              EXECUTE                        │
  │                 │                           ↓
  │                 └──────────────←────────────┘
  │
  └── mission complete ────→ END