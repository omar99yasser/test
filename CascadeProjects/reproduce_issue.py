
import json
import os
import sys
from typing import List, Dict, Optional

# Mock the GroqPromptingSystem class partially
class MockGroqSystem:
    def __init__(self):
        with open("openrouter_config.json", "r") as f:
            config = json.load(f)
        self.api_key = config.get("api_key")
        self.model = "llama-3.3-70b-versatile"
        
    def _call_groq(self, messages: List[Dict[str, str]]) -> str:
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
        }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _geocode_location(self, user_input: str, history: List[Dict[str, str]] = None) -> Optional[Dict[str, float]]:
        """Use LLM to extract coordinates from user input, using history for context."""
        
        # Get last user message for context if available
        last_user_msg = ""
        if history:
            for msg in reversed(history):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break
        
        system_prompt = f"""You are a geocoding assistant. Extract the reference location from the user's query and provide its approximate coordinates (latitude and longitude) in Los Angeles.
        
        Context - Previous User Query: "{last_user_msg}"
        Current User Query: "{user_input}"
        
        If the current query refers to "it", "the location", "there", or asks for a map without specifying a location, USE THE LOCATION FROM THE PREVIOUS QUERY.
        
        Output JSON ONLY: {{"latitude": float, "longitude": float, "location_name": "string"}}
        
        Example: "nearest station to Hollywood Sign" -> {{"latitude": 34.1341, "longitude": -118.3215, "location_name": "Hollywood Sign"}}
        Example (Context: "Hollywood Sign"): "show me map" -> {{"latitude": 34.1341, "longitude": -118.3215, "location_name": "Hollywood Sign"}}
        
        If no specific location is mentioned or implied by context, return null."""
        
        print(f"DEBUG: System Prompt:\n{system_prompt}\n")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        try:
            response = self._call_groq(messages)
            print(f"DEBUG: LLM Response:\n{response}\n")
            
            # Clean up response if it contains markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            data = json.loads(response)
            if data and "latitude" in data and "longitude" in data:
                return data
        except Exception as e:
            print(f"Geocoding error: {e}")
        
        return None

def main():
    system = MockGroqSystem()
    
    # Simulate history
    history = [
        {"role": "user", "content": "what is the nearest charging station to the Hollywood Walk of Fame"},
        {"role": "assistant", "content": "The nearest station is LADWP..."}
    ]
    
    current_query = "can you show it to me in map"
    
    print(f"Testing query: '{current_query}' with history...")
    result = system._geocode_location(current_query, history)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
