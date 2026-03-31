import sys
sys.path.insert(0, '/hpc/puhome/25116696g/NLP/story_maker')

print("Step 1: Import CoreferenceResolver")
try:
    from src.nlu.coreference import CoreferenceResolver
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print("Step 2: Create CoreferenceResolver instance")
try:
    coref = CoreferenceResolver()
    print("✅ Instance created")
except Exception as e:
    print(f"❌ Instance creation failed: {e}")
    sys.exit(1)

print("Step 3: Load model")
try:
    coref.load()
    print("✅ Load method called")
except Exception as e:
    print(f"❌ Load method failed: {e}")
    sys.exit(1)

print("Step 4: Check if model is loaded")
if coref.model is not None:
    print("✅ FastCoref model is loaded")
else:
    print("❌ FastCoref model is None")