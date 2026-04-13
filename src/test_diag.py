import sys
sys.path.append("src")
from src.vehicle_diagnostics import VehicleDiagnostics, SensorReading, SystemComponent, simulate_vehicle_startup

print("=== VEHICLE STARTUP CHECK ===")
diag = VehicleDiagnostics()
status = simulate_vehicle_startup(diag)
print("Status:", status["overall"])
print("Safe to drive:", status["safe_to_drive"])

print("\n=== SIMULATING FUEL LEAK ===")
diag.clear_alerts()
diag.process_reading(SensorReading(SystemComponent.FUEL, 1.5, "bar"))
status = diag.get_vehicle_status()
for a in status["alerts"]:
    print(f"[{a['level']}] {a['message']}")
    print(f"ACTION: {a['action']}")

print("\n=== SIMULATING BRAKE FAILURE ===")
diag.clear_alerts()
diag.process_reading(SensorReading(SystemComponent.BRAKES, 15.0, "%"))
status = diag.get_vehicle_status()
for a in status["alerts"]:
    print(f"[{a['level']}] {a['message']}")
    print(f"ACTION: {a['action']}")

print("\n=== SIMULATING LOW BATTERY ===")
diag.clear_alerts()
diag.process_reading(SensorReading(SystemComponent.BATTERY, 11.0, "V"))
status = diag.get_vehicle_status()
for a in status["alerts"]:
    print(f"[{a['level']}] {a['message']}")
    print(f"ACTION: {a['action']}")
