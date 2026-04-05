import sys
print("Python version:", sys.version)
print("Importing fastcoref...")
try:
    from fastcoref import FCoref
    print("fastcoref imported")
except Exception as e:
    print("Import error:", e)
    sys.exit(1)

print("Creating FCoref instance...")
try:
    model = FCoref(device="cpu")
    print("Model created:", model)
except Exception as e:
    print("Model creation error:", e)
    sys.exit(1)

print("Done")