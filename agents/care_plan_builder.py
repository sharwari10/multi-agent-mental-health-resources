import json
from agents.xai_config import create_chat_completion_with_fallback, create_xai_client

def run_care_plan_builder(web_results: str, kb_data: str, urgency: str) -> str:
    """
    Agent 3: Care Plan Builder Agent
    Generates structured JSON care plan based on web results and KB data.
    """
    client = create_xai_client()
    
    system_instruction = f"""
    You are the Care Plan Builder Agent.
    Based on the provided web search results and local knowledge base data, construct a structured care plan.
    You must reason based on the user's urgency ({urgency}).
    If urgency is High or Immediate Crisis, immediate resources MUST include crisis lines.
    Output strictly as JSON conforming to the following schema:
    {{
      "immediate": ["string array of immediate actions or crisis resources"],
      "short_term": ["string array of short term steps to take within a few days/weeks"],
      "long_term": ["string array of long term habits, therapy, or management strategies"]
    }}
    Do not output anything other than the JSON object. Do not wrap it in markdown block.
    """

    prompt = f"""
    Web Results:
    {web_results}
    
    Local KB Data:
    {kb_data}
    
    Urgency: {urgency}
    
    Generate the care plan JSON.
    """

    try:
        response = create_chat_completion_with_fallback(
            client,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        fallback = {
            "immediate": ["If you are in crisis, please call 988 or go to the nearest emergency room immediately.", f"System Error: {e}"],
            "short_term": ["Schedule an appointment with a local mental health professional."],
            "long_term": ["Explore ongoing therapy and community support groups."]
        }
        return json.dumps(fallback)
