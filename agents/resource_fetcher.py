import json
import os
from agents.xai_config import create_chat_completion_with_fallback, create_xai_client

def run_resource_fetcher(concern: str, location: str, urgency: str) -> str:
    """
    Agent 2: Resource Fetcher Agent
    Queries the local JSON knowledge base using Grok function calling.
    """
    client = create_xai_client()
    
    def query_local_kb(categories: list[str]) -> str:
        try:
            kb_path = os.path.join(os.path.dirname(__file__), "..", "data", "mental_health_kb.json")
            with open(kb_path, "r") as f:
                kb_data = json.load(f)
            
            results = {}
            for cat in categories:
                if cat in kb_data:
                    results[cat] = kb_data[cat]
            return json.dumps(results)
        except Exception as e:
            return json.dumps({"error": f"Failed to read KB: {e}"})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_local_kb",
                "description": "Queries the local knowledge base for specific categories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["CRISIS_LINES", "THERAPY_PLATFORMS", "DIGITAL_TOOLS", "COMMUNITY_SUPPORT", "HELPLINES"]
                            },
                            "description": "List of categories to fetch."
                        }
                    },
                    "required": ["categories"]
                }
            }
        }
    ]

    system_instruction = f"""
    You are the Resource Fetcher Agent.
    Your task is to identify which categories of local resources are needed based on the user's concern, location, and urgency.
    You must call the query_local_kb tool to fetch the data.
    If urgency is High or Immediate Crisis, you MUST request CRISIS_LINES.
    After fetching, output the JSON data of the relevant resources exactly as returned by the tool. Do not add markdown formatting, just output valid JSON.
    
    User Info:
    - Concern: {concern}
    - Location: {location}
    - Urgency: {urgency}
    """

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": "Fetch the appropriate local resources."}
    ]

    try:
        response = create_chat_completion_with_fallback(
            client,
            messages=messages,
            tools=tools,
            temperature=0.1
        )
        
        message = response.choices[0].message
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "query_local_kb":
                    args = json.loads(tool_call.function.arguments)
                    tool_result = query_local_kb(args["categories"])
                    
                    messages.append(message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_result
                    })
                    
                    final_response = create_chat_completion_with_fallback(
                        client,
                        messages=messages,
                        temperature=0.1
                    )
                    return final_response.choices[0].message.content
        return message.content
    except Exception as e:
        return json.dumps({"error": f"Resource Fetcher failed: {e}"})
