"""
src/vehicle_diagnostics.py
Vehicle health monitoring system.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    OK        = "OK"
    WARNING   = "WARNING"
    CRITICAL  = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class SystemComponent(Enum):
    ENGINE       = "engine"
    FUEL         = "fuel"
    BRAKES       = "brakes"
    TIRES        = "tires"
    BATTERY      = "battery"
    TEMPERATURE  = "temperature"
    OIL          = "oil"
    TRANSMISSION = "transmission"


@dataclass
class SensorReading:
    component: SystemComponent
    value:     float
    unit:      str
    timestamp: float = field(default_factory=time.time)


@dataclass
class VehicleAlert:
    component: SystemComponent
    level:     AlertLevel
    message:   str
    value:     float
    threshold: float
    action:    str
    timestamp: float = field(default_factory=time.time)


UNIT_TO_METRIC = {
    "°C":  "temp_c",
    "RPM": "rpm",
    "%":   "level_pct",
    "bar": "pressure_bar",
    "PSI": "pressure_psi",
    "psi": "pressure_psi",
    "V":   "voltage_v",
}

NORMAL_RANGES = {
    SystemComponent.ENGINE: {
        "rpm":    {"min": 600,  "max": 6000, "warning": 5500, "critical": 6000, "lower_is_worse": False},
        "temp_c": {"min": 70,   "max": 110,  "warning": 100,  "critical": 110,  "lower_is_worse": False},
    },
    SystemComponent.FUEL: {
        "level_pct":    {"min": 5,   "max": 100, "warning": 15,  "critical": 5,   "lower_is_worse": True},
        "pressure_bar": {"min": 3.0, "max": 4.5, "warning": 2.5, "critical": 2.0, "lower_is_worse": True},
    },
    SystemComponent.BRAKES: {
        "pad_pct":  {"min": 20, "max": 100, "warning": 30,  "critical": 20,  "lower_is_worse": True},
        "fluid_pct":{"min": 70, "max": 100, "warning": 75,  "critical": 70,  "lower_is_worse": True},
        "temp_c":   {"min": 0,  "max": 400, "warning": 350, "critical": 400, "lower_is_worse": False},
    },
    SystemComponent.TIRES: {
        "pressure_psi": {"min": 28, "max": 38, "warning": 30, "critical": 28, "lower_is_worse": True},
    },
    SystemComponent.BATTERY: {
        "voltage_v":  {"min": 11.5, "max": 14.8, "warning": 12.0, "critical": 11.5, "lower_is_worse": True},
        "charge_pct": {"min": 10,   "max": 100,  "warning": 20,   "critical": 10,   "lower_is_worse": True},
    },
    SystemComponent.OIL: {
        "pressure_bar": {"min": 1.0, "max": 5.0, "warning": 1.5, "critical": 1.0, "lower_is_worse": True},
        "level_pct":    {"min": 20,  "max": 100, "warning": 30,  "critical": 20,  "lower_is_worse": True},
        "temp_c":       {"min": 80,  "max": 135, "warning": 125, "critical": 135, "lower_is_worse": False},
    },
}

ALERT_ACTIONS = {
    SystemComponent.FUEL: {
        "level_pct": {
            AlertLevel.WARNING:   "Refuel soon — find nearest station",
            AlertLevel.CRITICAL:  "REFUEL IMMEDIATELY — risk of breakdown",
            AlertLevel.EMERGENCY: "PULL OVER SAFELY — fuel critically low",
        },
        "pressure_bar": {
            AlertLevel.WARNING:   "Fuel pressure low — check fuel pump",
            AlertLevel.CRITICAL:  "FUEL LEAK POSSIBLE — stop engine safely",
            AlertLevel.EMERGENCY: "STOP VEHICLE — possible fuel leak detected",
        },
    },
    SystemComponent.ENGINE: {
        "temp_c": {
            AlertLevel.WARNING:   "Engine warm — reduce speed",
            AlertLevel.CRITICAL:  "ENGINE OVERHEATING — pull over safely",
            AlertLevel.EMERGENCY: "STOP ENGINE NOW — severe overheating",
        },
        "rpm": {
            AlertLevel.WARNING:   "High RPM — ease off accelerator",
            AlertLevel.CRITICAL:  "CRITICAL RPM — risk of engine damage",
            AlertLevel.EMERGENCY: "ENGINE REDLINE — stop vehicle now",
        },
    },
    SystemComponent.BRAKES: {
        "pad_pct": {
            AlertLevel.WARNING:   "Brake pads wearing — service soon",
            AlertLevel.CRITICAL:  "BRAKE PADS CRITICAL — service immediately",
            AlertLevel.EMERGENCY: "BRAKE FAILURE RISK — do not drive",
        },
        "temp_c": {
            AlertLevel.WARNING:   "Brakes hot — reduce braking",
            AlertLevel.CRITICAL:  "BRAKE FADE RISK — stop and cool",
            AlertLevel.EMERGENCY: "BRAKE FIRE RISK — stop immediately",
        },
    },
    SystemComponent.BATTERY: {
        "voltage_v": {
            AlertLevel.WARNING:   "Battery weak — check charging system",
            AlertLevel.CRITICAL:  "BATTERY CRITICAL — risk of stall",
            AlertLevel.EMERGENCY: "BATTERY FAILURE — pull over safely",
        },
    },
    SystemComponent.OIL: {
        "pressure_bar": {
            AlertLevel.WARNING:   "Oil pressure low — check oil level",
            AlertLevel.CRITICAL:  "LOW OIL PRESSURE — stop engine soon",
            AlertLevel.EMERGENCY: "STOP ENGINE — oil pressure failure",
        },
    },
}


def _determine_level(value: float, thresholds: dict) -> AlertLevel:
    critical       = thresholds.get("critical")
    warning        = thresholds.get("warning")
    min_val        = thresholds.get("min")
    max_val        = thresholds.get("max")
    lower_is_worse = thresholds.get("lower_is_worse", True)

    if critical is None or warning is None:
        return AlertLevel.OK

    # value is in normal range — no alert
    if min_val is not None and max_val is not None:
        if min_val <= value <= max_val:
            return AlertLevel.OK

    # value outside normal range
    if lower_is_worse:
        if value <= critical:  return AlertLevel.EMERGENCY
        if value <= warning:   return AlertLevel.CRITICAL
        return AlertLevel.WARNING
    else:
        if value >= critical:  return AlertLevel.EMERGENCY
        if value >= warning:   return AlertLevel.CRITICAL
        return AlertLevel.WARNING


class VehicleDiagnostics:

    def __init__(self):
        self.alerts:  List[VehicleAlert] = []
        self.history: List[Dict]         = []

    def process_reading(self,
                        reading: SensorReading) -> Optional[VehicleAlert]:
        component  = reading.component
        ranges     = NORMAL_RANGES.get(component, {})
        metric_key = UNIT_TO_METRIC.get(reading.unit)

        if metric_key and metric_key in ranges:
            thresholds = ranges[metric_key]
        else:
            thresholds = None
            metric_key = None
            for m, t in ranges.items():
                lo = t.get("min", 0)
                hi = t.get("max", 999999)
                if lo / 2 <= reading.value <= hi * 2:
                    thresholds = t
                    metric_key = m
                    break

        if thresholds is None:
            return None

        level = _determine_level(reading.value, thresholds)
        if level == AlertLevel.OK:
            return None

        action = (ALERT_ACTIONS
                  .get(component, {})
                  .get(metric_key, {})
                  .get(level, "Check vehicle immediately"))

        alert = VehicleAlert(
            component=component,
            level=level,
            message=(f"{component.value.upper()} {metric_key} "
                     f"= {reading.value}{reading.unit}"),
            value=reading.value,
            threshold=thresholds.get("warning", 0),
            action=action,
        )
        self.alerts.append(alert)
        logger.warning(f"[{level.value}] {alert.message} → {action}")
        return alert

    def process_batch(self, readings) -> list:
        return [a for r in readings
                for a in [self.process_reading(r)] if a]

    def get_vehicle_status(self) -> dict:
        if not self.alerts:
            return {"overall": "OK", "alerts": [], "safe_to_drive": True}
        levels = [a.level for a in self.alerts]
        if AlertLevel.EMERGENCY in levels:
            overall, safe = "EMERGENCY", False
        elif AlertLevel.CRITICAL in levels:
            overall, safe = "CRITICAL", False
        elif AlertLevel.WARNING in levels:
            overall, safe = "WARNING", True
        else:
            overall, safe = "OK", True
        return {
            "overall":       overall,
            "safe_to_drive": safe,
            "alert_count":   len(self.alerts),
            "alerts": [
                {"component": a.component.value,
                 "level":     a.level.value,
                 "message":   a.message,
                 "action":    a.action}
                for a in self.alerts
            ],
        }

    def clear_alerts(self):
        self.alerts = []


def simulate_vehicle_startup(diagnostics: "VehicleDiagnostics") -> dict:
    startup_readings = [
        SensorReading(SystemComponent.ENGINE,  75.0,  "°C"),
        SensorReading(SystemComponent.ENGINE,  850.0, "RPM"),
        SensorReading(SystemComponent.FUEL,    80.0,  "%"),
        SensorReading(SystemComponent.FUEL,    3.8,   "bar"),
        SensorReading(SystemComponent.BRAKES,  85.0,  "%"),
        SensorReading(SystemComponent.TIRES,   34.0,  "PSI"),
        SensorReading(SystemComponent.BATTERY, 12.6,  "V"),
        SensorReading(SystemComponent.OIL,     3.5,   "bar"),
    ]
    diagnostics.process_batch(startup_readings)
    return diagnostics.get_vehicle_status()


if __name__ == "__main__":
    diag = VehicleDiagnostics()

    print("\n=== VEHICLE STARTUP CHECK ===")
    status = simulate_vehicle_startup(diag)
    print("Status:", status["overall"])
    print("Safe to drive:", status["safe_to_drive"])

    print("\n=== SIMULATING FUEL LEAK ===")
    diag.clear_alerts()
    diag.process_reading(
        SensorReading(SystemComponent.FUEL, 1.5, "bar"))
    status = diag.get_vehicle_status()
    for a in status["alerts"]:
        print("[" + a["level"] + "] " + a["message"])
        print("ACTION: " + a["action"])

    print("\n=== SIMULATING BRAKE FAILURE ===")
    diag.clear_alerts()
    diag.process_reading(
        SensorReading(SystemComponent.BRAKES, 15.0, "%"))
    status = diag.get_vehicle_status()
    for a in status["alerts"]:
        print("[" + a["level"] + "] " + a["message"])
        print("ACTION: " + a["action"])

    print("\n=== SIMULATING LOW BATTERY ===")
    diag.clear_alerts()
    diag.process_reading(
        SensorReading(SystemComponent.BATTERY, 11.0, "V"))
    status = diag.get_vehicle_status()
    for a in status["alerts"]:
        print("[" + a["level"] + "] " + a["message"])
        print("ACTION: " + a["action"])