from src.nlu.coreference import CoreferenceResolver

def main():
    print("Testing CoreferenceResolver...")
    try:
        coref = CoreferenceResolver()
        coref.load()
        if coref.model is not None:
            print("✅ FastCoref LOADED successfully!")
        else:
            print("❌ FastCoref model is None, failed to load.")
    except Exception as e:
        print(f"❌ Exception occurred: {e}")

if __name__ == "__main__":
    main()
