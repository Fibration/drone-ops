# drone-ops
Agentic Multi-UAV Mission Control &amp; Incident Recovery

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        fleet_monitor(fleet_monitor)
        mission_commander(mission_commander)
        mission_planner(mission_planner)
        safety_reviewer(safety_reviewer)
        human_reviewer(human_reviewer)
        __end__([<p>__end__</p>]):::last
        __start__ --> fleet_monitor;
        fleet_monitor --> mission_commander;
        human_reviewer -. &nbsp;approve&nbsp; .-> __end__;
        human_reviewer -. &nbsp;revise&nbsp; .-> mission_planner;
        mission_commander -. &nbsp;finish&nbsp; .-> __end__;
        mission_commander -.-> fleet_monitor;
        mission_commander -.-> mission_planner;
        mission_commander -.-> safety_reviewer;
        mission_planner --> mission_commander;
        safety_reviewer -. &nbsp;approve&nbsp; .-> __end__;
        safety_reviewer -.-> human_reviewer;
        safety_reviewer -. &nbsp;revise&nbsp; .-> mission_planner;
```
