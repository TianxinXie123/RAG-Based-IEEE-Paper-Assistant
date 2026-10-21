# src/llm_engine.py
"""
LLM backends for answer generation.

Supported providers:
  - "claude": Anthropic Claude via the `anthropic` SDK       (env var: ANTHROPIC_API_KEY)
  - "gemini": Google Gemini via the `google-genai` SDK      (env var: GEMINI_API_KEY)

Each SDK is imported only when its provider is used, so you only need the
package for the provider you actually run.

Three kinds of requests are supported:
  - get_llm_response():      answer a question from a few retrieved chunks (RAG), with chat history
  - rewrite_query():         turn a follow-up question into a standalone one before retrieval
  - get_document_response(): run a task (summary, suggestions) over the WHOLE paper
"""
import base64
import os

PROVIDERS = {
    "claude": {"env_var": "ANTHROPIC_API_KEY", "default_model": "claude-haiku-5-5", "package": "anthropic"},
    "gemini": {"env_var": "GEMINI_API_KEY", "default_model": "gemini-2.5-flash", "package": "google-genai"},
}
DEFAULT_PROVIDER = "claude"

QA_INSTRUCTION = (
    "You are a helpful research assistant. Answer the user's question based ONLY on the "
    "provided research paper snippets. If the answer is not in the context, say that you "
    "don't know. Keep the answer professional, concise, and structured. "
    "When you use a snippet, cite it as [Source N]. Always answer in English."
)

DOCUMENT_INSTRUCTION = (
    "You are an experienced engineering reviewer reading an IEEE-style research paper. "
    "The full text of the paper is provided; it was extracted from a PDF. Figures appear only "
    "as their captions and tables as Markdown, both listed after the body text, and equations "
    "may be garbled. Base everything you say on the paper itself, and say so explicitly when "
    "the paper does not provide something. Always answer in English."
)

SUMMARY_TASK = (
    "Write a structured summary of this paper with these sections:\n"
    "1. Problem and motivation\n"
    "2. Proposed method or system (key components and how they work)\n"
    "3. Experimental setup and main results (include concrete numbers where given)\n"
    "4. Main contributions\n"
    "5. Limitations stated by the authors\n"
    "Keep it under about 500 words."
)

SUGGESTIONS_TASK = (
    "Act as a critical but constructive reviewer of this paper and the project it describes. Give:\n"
    "1. Strengths (2-4 bullet points)\n"
    "2. Weaknesses or open questions (e.g. missing baselines, unclear assumptions, "
    "reproducibility, scalability, evaluation gaps)\n"
    "3. Concrete, actionable suggestions to improve the work or extend the project, "
    "ordered by expected impact, each with a one-line justification\n"
    "Point to the specific section or claim of the paper each comment refers to."
)

REWRITE_INSTRUCTION = (
    "You rewrite follow-up questions about a research paper into standalone questions. "
    "Use the conversation to resolve references such as 'it', 'that method' or 'and on CIFAR10?'. "
    "If the question is already standalone, return it unchanged. "
    "Reply with the rewritten question only, in English, on a single line."
)

SYMBOL_INSTRUCTION = (
    "You read single characters from images of printed PDF text, exactly as printed. "
    "Reply with the character only, with no explanation."
)

# Room for adaptive thinking plus the visible answer
MAX_OUTPUT_TOKENS = 16000

# Earlier answers are shortened to this many characters when used as context for a rewrite
HISTORY_ANSWER_CHARS = 600


class LLMError(Exception):
    """Raised for any problem talking to the LLM (missing key, network, rate limit, ...)."""


def build_user_prompt(query, context_chunks):
    context_text = "\n\n".join(f"Source {i + 1}: {chunk}" for i, chunk in enumerate(context_chunks))
    return (
        "--- CONTEXT FROM PAPER ---\n"
        f"{context_text}\n\n"
        "--- USER QUESTION ---\n"
        f"{query}"
    )


# Clients are created once per (provider, api_key) and reused across questions.
_clients = {}


def _get_api_key(provider):
    env_var = PROVIDERS[provider]["env_var"]
    key = os.getenv(env_var)
    if not key:
        raise LLMError(f"{env_var} is not set. Set it in your environment before using provider '{provider}'.")
    return key


def _history_text(history):
    """Previous turns as plain text (answers shortened, they can be long)."""
    lines = []
    for question, answer in history:
        short = answer if len(answer) <= HISTORY_ANSWER_CHARS else answer[:HISTORY_ANSWER_CHARS] + " ..."
        lines.append(f"User: {question}\nAssistant: {short}")
    return "\n\n".join(lines)


def _gemini_generate(model, system, user_parts, history, on_text):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise LLMError("The 'google-genai' package is not installed. Run: pip install google-genai")

    api_key = _get_api_key("gemini")
    client = _clients.get(("gemini", api_key))
    if client is None:
        client = _clients[("gemini", api_key)] = genai.Client(api_key=api_key)

    if any(isinstance(part, bytes) for part, _ in user_parts):
        raise LLMError("Image input is only implemented for the 'claude' provider.")
    prompt = "\n\n".join(text for text, _ in user_parts)
    if history:
        prompt = f"--- CONVERSATION SO FAR ---\n{_history_text(history)}\n\n{prompt}"
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    if not response.text:
        raise LLMError("Gemini returned an empty response (it may have been blocked by safety filters).")
    if on_text:
        on_text(response.text)
    return response.text


def _claude_generate(model, system, user_parts, history, on_text):
    try:
        import anthropic
    except ImportError:
        raise LLMError("The 'anthropic' package is not installed. Run: pip install anthropic")

    api_key = _get_api_key("claude")
    client = _clients.get(("claude", api_key))
    if client is None:
        client = _clients[("claude", api_key)] = anthropic.Anthropic(api_key=api_key)

    # A part marked cacheable (the full paper text) is stored in Claude's prompt cache,
    # so asking for a summary and then suggestions on the same paper only pays for it once.
    # A part given as bytes is a PNG image.
    content = []
    for part, cacheable in user_parts:
        if isinstance(part, bytes):
            block = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.standard_b64encode(part).decode("ascii")}}
        else:
            block = {"type": "text", "text": part}
        if cacheable:
            block["cache_control"] = {"type": "ephemeral"}
        content.append(block)

    # Earlier turns are sent as plain question/answer pairs (without their retrieved context)
    messages = []
    for question, answer in history or []:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": content})

    try:
        # Streaming shows the answer as it is generated instead of after a long pause
        with client.messages.stream(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=messages,
        ) as stream:
            if on_text:
                for delta in stream.text_stream:
                    on_text(delta)
            message = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise LLMError("Claude rejected the API key. Check ANTHROPIC_API_KEY.")
    except anthropic.NotFoundError:
        raise LLMError(f"Unknown Claude model '{model}'. Run list_models.py to list the models your key can use.")
    except anthropic.RateLimitError:
        raise LLMError("Claude rate limit reached. Wait a moment and try again.")
    except anthropic.APIConnectionError:
        raise LLMError("Could not reach the Claude API. Check your internet connection.")
    except anthropic.APIStatusError as e:
        raise LLMError(f"Claude API error {e.status_code}: {e.message}")

    if message.stop_reason == "refusal":
        raise LLMError("Claude declined to answer this request.")

    # Keep only the text blocks of the reply (thinking blocks are skipped)
    text = "".join(block.text for block in message.content if block.type == "text")
    if not text:
        raise LLMError("Claude returned an empty response.")
    if message.stop_reason == "max_tokens":
        warning = "\n\n[WARNING] The answer was cut off because it reached the output token limit."
        text += warning
        if on_text:
            on_text(warning)
    return text


def _generate(provider, model, system, user_parts, history=None, on_text=None):
    """
    Dispatch to the provider and turn any unexpected exception into LLMError.
    `user_parts` is a list of (text, cacheable) pairs; `on_text` (optional) is called
    with each piece of the answer as it arrives.
    """
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    model = model or PROVIDERS[provider]["default_model"]

    try:
        if provider == "gemini":
            return _gemini_generate(model, system, user_parts, history, on_text)
        return _claude_generate(model, system, user_parts, history, on_text)
    except LLMError:
        raise
    except Exception as e:
        # Network errors, invalid key, rate limits, unknown model name, etc.
        raise LLMError(f"{provider} request failed ({type(e).__name__}): {e}") from e


def rewrite_query(query, history, provider=DEFAULT_PROVIDER, model=None):
    """
    Turn a follow-up question ("and on CIFAR10?") into a standalone question for retrieval.
    Returns the query unchanged if there is no history or the rewrite fails.
    """
    if not history:
        return query
    prompt = (f"--- CONVERSATION SO FAR ---\n{_history_text(history)}\n\n"
              f"--- FOLLOW-UP QUESTION ---\n{query}")
    try:
        rewritten = _generate(provider, model, REWRITE_INSTRUCTION, [(prompt, False)])
    except LLMError:
        return query
    # Guard against a chatty reply: keep the first non-empty line only
    first_line = next((line.strip() for line in rewritten.splitlines() if line.strip()), "")
    return first_line or query


def get_llm_response(query, context_chunks, provider=DEFAULT_PROVIDER, model=None, history=None, on_text=None):
    """
    Send the query + retrieved context (and earlier turns, if any) to the chosen LLM
    and return the answer text.
    Raises LLMError on any failure, so the caller can report it and keep running.
    """
    user_prompt = build_user_prompt(query, context_chunks)
    return _generate(provider, model, QA_INSTRUCTION, [(user_prompt, False)], history, on_text)


def identify_symbol(png_bytes, extracted_char, line_text, model=None):
    """
    Ask Claude which character is printed inside the red box of a line image.
    Used to repair characters that a PDF maps to the wrong Unicode text. Raises LLMError on failure.
    """
    question = (
        f"This image is one line of a PDF. Its text layer reads: {line_text!r}\n"
        f"The character inside the red box was extracted as {extracted_char!r}, which is probably wrong.\n"
        "Which character is printed inside the red box? Reply with that one Unicode character only "
        "(for example ε, ≤ or •). If it is an accent mark printed over the following letter, "
        "reply with the accent alone (for example ¨). Report what is printed, even if it looks like a typo."
    )
    return _generate("claude", model, SYMBOL_INSTRUCTION, [(png_bytes, False), (question, False)])


def get_document_response(task, full_text, provider=DEFAULT_PROVIDER, model=None, on_text=None):
    """
    Run a whole-document task (e.g. SUMMARY_TASK, SUGGESTIONS_TASK) over the full paper text.
    Raises LLMError on any failure.
    """
    paper = f"--- FULL TEXT OF THE PAPER ---\n{full_text}\n--- END OF PAPER ---"
    return _generate(provider, model, DOCUMENT_INSTRUCTION,
                     [(paper, True), (f"--- TASK ---\n{task}", False)], on_text=on_text)
