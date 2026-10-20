# list_models.py
# List the models your API key can use, for Gemini and/or Claude.
#   python list_models.py            -> checks every provider whose API key is set
import os

if os.getenv("GEMINI_API_KEY"):
    print("Gemini models (generateContent):")
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        for m in client.models.list():
            if "generateContent" in (m.supported_actions or []):
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("GEMINI_API_KEY not set, skipping Gemini.")

if os.getenv("ANTHROPIC_API_KEY"):
    print("\nClaude models:")
    try:
        import anthropic
        client = anthropic.Anthropic()
        for m in client.models.list():
            print(f"- {m.id}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("ANTHROPIC_API_KEY not set, skipping Claude.")
