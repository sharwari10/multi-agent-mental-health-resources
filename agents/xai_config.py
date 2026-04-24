import os
from openai import OpenAI


def create_xai_client() -> OpenAI:
    """
    Backward-compatible client factory used by all agents.
    Supports provider switch via .env:
      - LLM_PROVIDER=openai with OPENAI_API_KEY
      - LLM_PROVIDER=xai with XAI_API_KEY
    """
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "xai"

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        return OpenAI(api_key=api_key, base_url=base_url)

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY is not set.")
    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


def _model_candidates() -> list[str]:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "xai"

    if provider == "openai":
        primary = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        raw_fallbacks = os.getenv("OPENAI_MODEL_FALLBACKS", "gpt-4.1-mini,gpt-4.1")
    else:
        primary = os.getenv("XAI_MODEL", "grok-4-0709")
        raw_fallbacks = os.getenv(
            "XAI_MODEL_FALLBACKS",
            "grok-4.20-0309-reasoning,grok-4.20-0309-non-reasoning,grok-4-1-fast-reasoning,grok-3,grok-3-mini,grok-code-fast-1,grok-2-vision-1212",
        )
    fallbacks = [m.strip() for m in raw_fallbacks.split(",") if m.strip()]

    candidates: list[str] = []
    for model in [primary, *fallbacks]:
        if model and model not in candidates:
            candidates.append(model)
    return candidates


def _available_models(client: OpenAI) -> list[str]:
    try:
        models = client.models.list()
        ids = [m.id for m in models.data if getattr(m, "id", None)]
        # Prioritize Grok models, then keep all others as a final fallback.
        grok_ids = [m for m in ids if "grok" in m.lower()]
        other_ids = [m for m in ids if m not in grok_ids]
        return [*grok_ids, *other_ids]
    except Exception:
        return []


def create_chat_completion_with_fallback(client: OpenAI, **kwargs):
    last_error = None
    configured = _model_candidates()
    discovered = _available_models(client)

    candidates: list[str] = []
    for model in [*configured, *discovered]:
        if model and model not in candidates:
            candidates.append(model)

    for model in candidates:
        try:
            return client.chat.completions.create(model=model, **kwargs)
        except Exception as exc:
            message = str(exc)
            if (
                "Incorrect API key" in message
                or "invalid_api_key" in message
                or "AuthenticationError" in message
            ):
                raise RuntimeError(
                    "LLM authentication failed. "
                    "Check OPENAI_API_KEY (if LLM_PROVIDER=openai) or "
                    "XAI_API_KEY (if LLM_PROVIDER=xai)."
                ) from exc
            if "Model not found" in message or "invalid argument" in message:
                last_error = exc
                continue
            raise

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    raise RuntimeError(
        "All configured models failed. "
        f"Tried: {', '.join(candidates)}. "
        f"Last error: {last_error}. "
        "Set OPENAI_MODEL or XAI_MODEL in .env to a model ID enabled for your provider."
    )
