import os
import json
from tavily import TavilyClient
from agents.xai_config import create_chat_completion_with_fallback, create_xai_client

def run_concern_analyser(concern: str, location: str, age: str, urgency: str) -> str:
    """
    Agent 1: Concern Analyser Agent
    Uses Tavily Search API with an agentic loop via Grok tool calling.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
    
    client = create_xai_client()
    
    def search_web(query: str) -> str:
        if not tavily_client:
            return "Tavily API key not found. Cannot search the web."
        try:
            response = tavily_client.search(query=query, max_results=3)
            return str(response)
        except Exception as e:
            return f"Search failed: {e}"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Searches the web for mental health resources, crisis helplines, therapists, apps, or support groups based on the query. Provide location in query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    system_instruction = f"""
    You are the Concern Analyser Agent.
    Your task is to analyze the user's mental health concern, location, age, and urgency.
    You must use the search_web tool to find relevant resources: crisis helplines, therapists, apps, or support groups.
    If the urgency is high, prioritize crisis helplines.
    Gather information and provide a structured summary of the resources found.
    Do not stop searching until you have at least 2 good resources, but do not exceed 12 iterations.
    Make sure to search specifically for resources near the user's location.
    
    User Info:
    - Concern: {concern}
    - Location: {location}
    - Age: {age}
    - Urgency: {urgency}
    """

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": "Begin analysis and search."}
    ]

    try:
        iterations = 0
        while iterations < 12:
            response = create_chat_completion_with_fallback(
                client,
                messages=messages,
                tools=tools,
                temperature=0.2
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "search_web":
                        args = json.loads(tool_call.function.arguments)
                        tool_result = search_web(args["query"])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": tool_result
                        })
            else:
                return message.content
            
            iterations += 1
            
        return messages[-1].content if messages[-1].content else "Max iterations reached."
        
    except Exception as e:
        return f"Concern Analyser failed: {e}"
