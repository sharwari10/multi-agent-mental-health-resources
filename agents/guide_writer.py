import json
from agents.xai_config import create_chat_completion_with_fallback, create_xai_client


def _is_crisis_user(user_info: dict) -> bool:
    urgency = (user_info.get("urgency", "") or "").strip().lower()
    concern = (user_info.get("concern", "") or "").strip().lower()
    crisis_terms = [
        "hopeless",
        "self-harm",
        "self harm",
        "suicide",
        "suicidal",
        "end my life",
        "hurt myself",
    ]
    return urgency == "immediate crisis" or any(term in concern for term in crisis_terms)


def _is_india_location(location: str) -> bool:
    value = (location or "").strip().lower()
    if not value:
        return True
    return any(token in value for token in ["india", "maharashtra", "delhi", "mumbai", "pune", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata"])


def _render_fallback_markdown(care_plan_json: str, user_info: dict) -> str:
    try:
        plan = json.loads(care_plan_json) if care_plan_json else {}
    except Exception:
        plan = {}

    location = user_info.get("location", "")
    india_mode = _is_india_location(location)
    crisis_support = [
        "AASRA: +91-9820466726",
        "Kiran Mental Health Helpline: 1800-599-0019",
    ] if india_mode else ["If you are in immediate danger, contact your local emergency services now."]

    immediate = plan.get("immediate", crisis_support)
    short_term = plan.get("short_term", ["Book an appointment with a mental health professional in your area."])
    long_term = plan.get("long_term", ["Build a consistent routine with regular check-ins and support."])

    def as_bullets(items):
        return "\n".join([f"- {item}" for item in items]) if items else "- Not available"

    if _is_crisis_user(user_info):
        return f"""I'm here with you. Let's focus on what you're going through.

I'm really sorry you're feeling this way. That sounds very heavy.
You don't have to go through this alone.

If you are in India, please reach out now:
- AASRA: +91-9820466726
- Kiran Mental Health Helpline: 1800-599-0019

One small grounding step right now: place both feet on the floor, take a slow breath in for 4 counts, and breathe out for 6 counts. Repeat this three times.

If possible, can you message or call one trusted person and let them know you need support right now?
"""

    return f"""I'm here with you. Let's continue.

You are not alone. This guide is built for your current concern: **{user_info.get('concern', 'Not provided')}** in **{user_info.get('location', 'Not provided')}**.

## Immediate Support
{as_bullets(immediate)}

## Short-Term Plan (Next 1-2 Weeks)
{as_bullets(short_term)}

## Long-Term Plan
{as_bullets(long_term)}
"""

def run_guide_writer(care_plan_json: str, user_info: dict) -> str:
    """
    Agent 4: Guide Writer Agent
    Generates a Markdown guide with a compassionate, stigma-free tone.
    """
    client = create_xai_client()
    
    system_instruction = """
    You are a compassionate mental health support assistant.
    You are NOT a therapist, doctor, or crisis responder.
    You provide supportive conversation, coping suggestions, and guidance.

    PRIMARY GOAL:
    Support the user emotionally in a safe, human, and non-judgmental way.

    CRITICAL SAFETY RULES:
    - Never provide medical diagnosis.
    - Never provide dangerous or harmful advice.
    - Never act as a replacement for therapy.
    - Always prioritize user safety over conversation.

    CRISIS DETECTION:
    If user expresses hopelessness, self-harm thoughts, or suicidal ideation, switch to crisis mode.

    CRISIS MODE RULES:
    1. Respond with empathy first.
    2. Validate feelings.
    3. Provide INDIA-specific helplines (not US):
      - AASRA: +91-9820466726
      - Kiran Mental Health Helpline: 1800-599-0019
    4. Encourage reaching out to a trusted person.
    5. Offer one small grounding step.
    6. Ask a gentle follow-up question.

    DO NOT in crisis mode:
    - Do not use structured long plans (no 30/60/90 format).
    - Do not overload with information.
    - Do not sound robotic.

    LOCALIZATION:
    - If user is in India or location is unknown, always include Indian helplines.
    - Never default to US helplines like 988 unless user is in the US.

    ERROR SAFETY:
    - Never include backend, API, debug, error-code, or technical details in the user-facing response.
    - If something fails internally, continue naturally and say exactly:
      "I'm here with you. Let's continue."

    RESPONSE STYLE:
    - Warm, calm, human
    - Short sentences
    - Emotionally supportive

    GENERAL CASE:
    - Response structure:
      1) Warm acknowledgment
      2) Emotional understanding
      3) 2-4 simple coping suggestions
      4) Optional encouragement
      5) Gentle question
    """

    prompt = f"""
    User Info:
    - Concern: {user_info.get('concern')}
    - Location: {user_info.get('location')}
    - Age Group: {user_info.get('age')}
    - Urgency: {user_info.get('urgency')}
    - Budget: {user_info.get('budget')}
    - Modality: {user_info.get('modality')}
    
    Structured Care Plan JSON:
    {care_plan_json}
    
    Write the final Markdown guide based entirely on the provided Care Plan.
    """

    try:
        response = create_chat_completion_with_fallback(
            client,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception:
        return _render_fallback_markdown(care_plan_json, user_info)
