"""
tests/test_diagnostics.py
Tests for vehicle health monitoring and alert system.
"""

import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from vehicle_diagnostics import (
    VehicleDiagnostics, SensorReading,
    SystemComponent, AlertLevel,
    simulate_vehicle_startup,
)


class TestStartupCheck:
    def test_healthy_vehicle_passes_startup(self):
        diag = VehicleDiagnostics()
        status = simulate_vehicle_startup(diag)
        assert status["safe_to_drive"] is True

    def test_startup_returns_status(self):
        diag = VehicleDiagnostics()
        status = simulate_vehicle_startup(diag)
        assert "overall" in status
        assert "safe_to_drive" in status
        assert "alerts" in status


class TestFuelSystem:
    def test_low_fuel_warning(self):
        diag = VehicleDiagnostics()
        alert = diag.process_reading(
            SensorReading(SystemComponent.FUEL, 10.0, "%"))
        assert alert is not None
        assert alert.level in [AlertLevel.WARNING,
                                AlertLevel.CRITICAL,
                                AlertLevel.EMERGENCY]

    def test_fuel_leak_detected(self):
        diag = VehicleDiagnostics()
        alert = diag.process_reading(
            SensorReading(SystemComponent.FUEL, 1.5, "bar"))
        assert alert is not None
        assert alert.level in [AlertLevel.CRITICAL,
                                AlertLevel.EMERGENCY]

    def test_normal_fuel_no_alert(self):
        diag = VehicleDiagnostics()
        alert = diag.process_reading(
            SensorReading(SystemComponent.FUEL, 75.0, "%"))
        assert alert is None


class TestBrakeSystem:
    def test_worn_brake_pads_alert(self):
        diag = VehicleDiagnostics()
        alert = diag.process_reading(
            SensorReading(SystemComponent.BRAKES, 15.0, "%"))
        assert alert is not None
        assert alert.level == AlertLevel.EMERGENCY

    def test_overheated_brakes_alert(self):
        diag = VehicleDiagnostics()
        alert = diag.process_reading(
            SensorReading(SystemComponent.BRAKES, 420.0, "°C"))
        assert alert is not None

    def test_good_brakes_no_alert(self):
        diag = VehicleDiagnostics()
        alert = diag.process_reading(
            SensorReading(SystemComponent.BRAKES, 90.0, "%"))
        assert alert is None


class TestEngineSystem:
    def test_overheating_engine_alert(self):
        diag = VehicleDiagnostics()
        alert = diag.process_reading(
            SensorReading(SystemComponent.ENGINE, 115.0, "°C"))
        assert alert is not None
        assert alert.level in [AlertLevel.CRITICAL,
                                AlertLevel.EMERGENCY]

    def test_normal_engine_temp_ok(self):
        diag = VehicleDiagnostics()
        alert = diag.process_reading(
            SensorReading(SystemComponent.ENGINE, 90.0, "°C"))
        assert alert is None


class TestBatterySystem:
    def test_dead_battery_alert(self):
        diag = VehicleDiagnostics()
        alert = diag.process_reading(
            SensorReading(SystemComponent.BATTERY, 11.0, "V"))
        assert alert is not None
        assert alert.level in [AlertLevel.CRITICAL,
                                AlertLevel.EMERGENCY]

    def test_good_battery_no_alert(self):
        diag = VehicleDiagnostics()
        alert = diag.process_reading(
            SensorReading(SystemComponent.BATTERY, 12.6, "V"))
        assert alert is None


class TestAlertPriority:
    def test_emergency_overrides_warning(self):
        diag = VehicleDiagnostics()
        # Add warning first
        diag.process_reading(
            SensorReading(SystemComponent.FUEL, 10.0, "%"))
        # Add emergency
        diag.process_reading(
            SensorReading(SystemComponent.BRAKES, 15.0, "%"))
        status = diag.get_vehicle_status()
        assert status["overall"] == "EMERGENCY"
        assert status["safe_to_drive"] is False

    def test_alert_has_action_message(self):
        diag = VehicleDiagnostics()
        diag.process_reading(
            SensorReading(SystemComponent.FUEL, 3.0, "%"))
        status = diag.get_vehicle_status()
        for alert in status["alerts"]:
            assert len(alert["action"]) > 0

    def test_clear_alerts_works(self):
        diag = VehicleDiagnostics()
        diag.process_reading(
            SensorReading(SystemComponent.FUEL, 3.0, "%"))
        diag.clear_alerts()
        assert diag.get_vehicle_status()["overall"] == "OK"

    def test_batch_processing(self):
        diag = VehicleDiagnostics()
        readings = [
            SensorReading(SystemComponent.FUEL,    80.0, "%"),
            SensorReading(SystemComponent.ENGINE,  90.0, "°C"),
            SensorReading(SystemComponent.BATTERY, 12.6, "V"),
            SensorReading(SystemComponent.BRAKES,  15.0, "%"),
        ]
        alerts = diag.process_batch(readings)
        assert len(alerts) >= 1