import json
from agents.xai_config import create_chat_completion_with_fallback, create_xai_client

def _heuristic_judge(guide_markdown: str) -> str:
    text = (guide_markdown or "").lower()
    has_immediate = "immediate" in text or "emergency" in text or "crisis" in text
    has_short = "short-term" in text or "short term" in text
    has_long = "long-term" in text or "long term" in text
    has_goals = "30/60/90" in text or "30 days" in text

    scores = {
        "resource_relevance": 7 if len(text) > 200 else 5,
        "safety_and_crisis_handling": 8 if has_immediate else 5,
        "personalization": 6,
        "completeness": 8 if (has_short and has_long and has_goals) else 6,
        "tone": 7,
    }
    overall = round(sum(scores.values()) / len(scores), 1)

    fallback = {
        "scores": scores,
        "overall_score": overall,
        "summary": "Heuristic evaluation used because LLM judge was unavailable.",
        "top_strength": "Safety and structure coverage" if has_immediate else "Basic support structure",
        "top_improvement": "Add more local, provider-specific resources and personalization.",
        "safety_flag": "Manual safety review recommended (LLM judge unavailable).",
    }
    return json.dumps(fallback)

def run_judge_agent(guide_markdown: str, user_info: dict) -> str:
    """
    Agent 5: LLM-as-Judge Agent
    Evaluates the final guide in a separate context without sharing previous agent context.
    """
    client = create_xai_client()
    
    system_instruction = """
    You are an impartial LLM-as-Judge Agent evaluating a mental health resource guide.
    Evaluate the guide strictly on the following 5 criteria:
    - Resource relevance
    - Safety & crisis handling
    - Personalization
    - Completeness
    - Tone
    Provide structured JSON output strictly matching this schema:
    {
      "scores": {
        "resource_relevance": <int 1-10>,
        "safety_and_crisis_handling": <int 1-10>,
        "personalization": <int 1-10>,
        "completeness": <int 1-10>,
        "tone": <int 1-10>
      },
      "overall_score": <float>,
      "summary": "<string>",
      "top_strength": "<string>",
      "top_improvement": "<string>",
      "safety_flag": "<string, empty if none>"
    }
    Do not output any markdown block formatting, just the raw JSON object.
    """

    prompt = f"""
    User Info:
    {json.dumps(user_info, indent=2)}
    
    Final Guide to Evaluate:
    {guide_markdown}
    """

    try:
        response = create_chat_completion_with_fallback(
            client,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception:
        return _heuristic_judge(guide_markdown)
