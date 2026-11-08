from collections.abc import Sequence
from typing import Protocol

from src.models import (
    TelemetryCapability, TelemetryField, TelemetryReading, TelemetryRequest,
    TelemetryResult, TelemetrySnapshot,
)


Readings = dict[str, list[TelemetryReading]]


class TelemetryQueries(Protocol):
    def available_telemetry(self) -> list[TelemetryCapability]: ...
    def drone_ids(self) -> list[str]: ...
    def battery_level(self, drone_ids: Sequence[str] | str | None = None, *, at_step: int | None = None, history: bool = False) -> Readings: ...
    def position(self, drone_ids: Sequence[str] | str | None = None, *, at_step: int | None = None, history: bool = False) -> Readings: ...
    def comms(self, drone_ids: Sequence[str] | str | None = None, *, at_step: int | None = None, history: bool = False) -> Readings: ...
    def status(self, drone_ids: Sequence[str] | str | None = None, *, at_step: int | None = None, history: bool = False) -> Readings: ...
    def assignment(self, drone_ids: Sequence[str] | str | None = None, *, at_step: int | None = None, history: bool = False) -> Readings: ...


class RecordedTelemetry:
    """Read-only queries over recorded snapshots; no raw drone state is exposed."""

    def __init__(self, snapshots: Sequence[TelemetrySnapshot]):
        self._snapshots = snapshots

    def available_telemetry(self) -> list[TelemetryCapability]:
        descriptions = {
            TelemetryField.BATTERY_LEVEL: "Battery level in simulator energy units",
            TelemetryField.POSITION: "Zero-based (row, column) position",
            TelemetryField.COMMS: "Whether communication is available",
            TelemetryField.STATUS: "Reported drone status",
            TelemetryField.ASSIGNMENT: "Last reported assignment",
        }
        return [TelemetryCapability(field=field, description=text)
                for field, text in descriptions.items()]

    def drone_ids(self) -> list[str]:
        return list(dict.fromkeys(drone_id for snapshot in self._snapshots
                                  for drone_id in snapshot.drones))

    def _read(self, field: TelemetryField, drone_ids: Sequence[str] | str | None,
              at_step: int | None, history: bool) -> Readings:
        """Latest means last recorded reading at or before at_step, if specified."""
        if at_step is not None and at_step < 0:
            raise ValueError("at_step must be non-negative")
        available = self.drone_ids()
        selected = available if drone_ids is None else ([drone_ids] if isinstance(drone_ids, str) else list(drone_ids))
        unknown = set(selected) - set(available)
        if unknown:
            raise ValueError(f"Unknown drone IDs: {sorted(unknown)}")
        result: Readings = {drone_id: [] for drone_id in selected}
        attribute = "battery" if field == TelemetryField.BATTERY_LEVEL else field.value
        for snapshot in sorted(self._snapshots, key=lambda item: item.step):
            if at_step is not None and snapshot.step > at_step:
                break
            for drone_id in result:
                drone = snapshot.drones.get(drone_id)
                if drone is None:
                    continue
                if field != TelemetryField.COMMS and not drone.comms:
                    # A lost link cannot deliver fresh position/battery reports.
                    continue
                reading = TelemetryReading(drone_id=drone_id, step=snapshot.step,
                                           field=field, value=getattr(drone, attribute))
                if history:
                    result[drone_id].append(reading)
                else:
                    result[drone_id] = [reading]
        return result

    def battery_level(self, drone_ids: Sequence[str] | str | None = None, *, at_step: int | None = None, history: bool = False) -> Readings:
        return self._read(TelemetryField.BATTERY_LEVEL, drone_ids, at_step, history)

    def position(self, drone_ids: Sequence[str] | str | None = None, *, at_step: int | None = None, history: bool = False) -> Readings:
        return self._read(TelemetryField.POSITION, drone_ids, at_step, history)

    def comms(self, drone_ids: Sequence[str] | str | None = None, *, at_step: int | None = None, history: bool = False) -> Readings:
        return self._read(TelemetryField.COMMS, drone_ids, at_step, history)

    def status(self, drone_ids: Sequence[str] | str | None = None, *, at_step: int | None = None, history: bool = False) -> Readings:
        return self._read(TelemetryField.STATUS, drone_ids, at_step, history)

    def assignment(self, drone_ids: Sequence[str] | str | None = None, *, at_step: int | None = None, history: bool = False) -> Readings:
        return self._read(TelemetryField.ASSIGNMENT, drone_ids, at_step, history)


def merge_telemetry(previous: list[TelemetryResult], updates: list[TelemetryResult]) -> list[TelemetryResult]:
    results = {item.request.model_dump_json(): item for item in previous}
    results.update((item.request.model_dump_json(), item) for item in updates)
    return list(results.values())


def has_telemetry(results: list[TelemetryResult], request: TelemetryRequest) -> bool:
    return any(result.request == request for result in results)
