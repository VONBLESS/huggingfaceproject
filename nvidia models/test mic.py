import sounddevice as sd

print("Default device:", sd.default.device)
print("\nAvailable audio devices:")
print(sd.query_devices())
