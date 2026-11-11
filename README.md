# drone-ops
Agentic Multi-UAV Mission Control &amp; Incident Recovery

This repo is split between a simulator and an agent team. 

The simulator runs a discrete time UAV drone search and rescue simulation
on a discrete grid partitioned into sectors.
Each sector has its own terrain profile, weather, target population and
target survival profile reflecting its hazard level.
The global weather and challenge difficulty is determined by a scenario card.

The agent team is a harness for multi-agent cooperation.
The roles are fleet monitor, mission commander, mission planner, and reviewer.
Each role is provided with a set of informed tools whose dependencies are injected
with a simulator dispatcher.
The agents run on a LangGraph graph where the mission commander acts as a high level
posture setter and a router.
All inputs and outputs for agents are TypedDicts as guardrails.

- The mission commander sets priorities and delegates to the mission planner
- The mission planner produces up to 2 plans based on priorities
- The mission commander selects one of the returned plans
- The reviewer evaluates the appropriateness of the priorities, and stress tests the selected plan
- The commander has an opportunity to redirect to the mission planner in the case of an 
unfavourable review
- A disagreement between the mission commander and reviewer results in human review
- The fleet monitor decides what to monitor based on the plan which would next wake
the agent team

Each agent acts using a policy.
This repo provides OpenAI LLM policies as well as naive heuristic policies.

```mermaid
graph TD;
	MISSION_COMMANDER -.->|1| MISSION_PLANNER;
	MISSION_COMMANDER -.->|2| REVIEWER;
	MISSION_COMMANDER -.-> __end__;
	MISSION_COMMANDER -.->|B4| human_reviewer;
	MISSION_PLANNER --> MISSION_COMMANDER;
	REVIEWER -.->|A3| FLEET_MONITOR;
	REVIEWER -.->|B3| MISSION_COMMANDER;
	__start__ --> MISSION_COMMANDER;
	human_reviewer -.-> FLEET_MONITOR;
	human_reviewer -.-> MISSION_COMMANDER;
	FLEET_MONITOR --> __end__;
```

A world builder utility which can generate a random world from a scenario card
can be found in `scenario.py`.

Running OpenAI agents requires a `.env` file with `OPENAI_API_KEY=""`.