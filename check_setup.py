# check_setup.py - verify the environment: packages, embedding model, API keys.
# Never prints full API keys.
import os
import sys

ok = True
print(f"Python {sys.version.split()[0]} at {sys.executable}")

# 1. Packages
try:
    import faiss, numpy  # noqa: F401
    import pymupdf  # noqa: F401
    import sentence_transformers  # noqa: F401
    import anthropic
    from google import genai  # noqa: F401
    print("[OK] All packages import correctly")
except Exception as e:
    ok = False
    print(f"[FAIL] Package import: {type(e).__name__}: {e}")

# 2. Embedding model (downloads ~80MB the first time, so main.py starts fast later)
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"[OK] Embedding model ready, vector size {m.encode(['test']).shape[1]}")
except Exception as e:
    ok = False
    print(f"[FAIL] Embedding model: {type(e).__name__}: {e}")

# 3. Claude key
key = os.getenv("ANTHROPIC_API_KEY")
if not key:
    print("[SKIP] ANTHROPIC_API_KEY not set (run: setx ANTHROPIC_API_KEY \"sk-ant-...\" then reopen the window)")
else:
    print(f"[..] ANTHROPIC_API_KEY found ({key[:10]}..., {len(key)} chars)")
    try:
        client = anthropic.Anthropic()
        r = client.messages.create(
            model="claude-haiku-5-5", max_tokens=100,
            messages=[{"role": "user", "content": "Reply with exactly: Claude API key works!"}],
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        print(f"[OK] Claude replied: {text.strip()}")
    except Exception as e:
        ok = False
        print(f"[FAIL] Claude API call: {type(e).__name__}: {e}")

# 4. Gemini key (optional)
if os.getenv("GEMINI_API_KEY"):
    try:
        from google import genai
        r = genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models.generate_content(
            model="gemini-2.5-flash", contents="Reply with exactly: Gemini API key works!")
        print(f"[OK] Gemini replied: {(r.text or '').strip()}")
    except Exception as e:
        print(f"[WARN] Gemini API call failed (optional): {type(e).__name__}: {e}")
else:
    print("[SKIP] GEMINI_API_KEY not set (optional)")

sys.exit(0 if ok else 1)
