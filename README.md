# drone-ops
Agentic Multi-UAV Mission Control &amp; Incident Recovery

```mermaid
flowchart TD
    start([START]) --> fleetMonitor[FLEET MONITOR]
    fleetMonitor --> commander[COMMANDER]
    commander -->|telemetry needed| fleetMonitor
    commander -->|planning needed| planner[PLANNER]
    planner --> commander
    commander -->|plan ready| safetyReview[SAFETY REVIEW]
    safetyReview --> approve[APPROVE]
    safetyReview --> revise[REVISE]
    safetyReview --> humanApproval[HUMAN APPROVAL]
    approve --> execute[EXECUTE]
    revise --> planner
    humanApproval --> execute
    commander -->|mission complete| finish([END])
```
