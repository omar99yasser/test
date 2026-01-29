#!/usr/bin/env python3
"""
OpenRouter prompting system with 4D prompt agent framework and RAG support.

Features:
  * User and system prompt management
  * 4D prompt agent (Context, Intent, Behavior, Temporal dimensions)
  * OpenRouter API integration
  * RAG (Retrieval-Augmented Generation) for dataset queries
  * Configurable model selection
  * Conversation history management
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


class FourDPromptAgent:
    """4D Prompt Agent: Context, Intent, Behavior, Temporal dimensions."""

    def __init__(self):
        self.context_history: List[Dict[str, str]] = []
        self.user_intent_patterns: Dict[str, int] = {}
        self.behavior_metrics: Dict[str, Any] = {}
        self.temporal_context: Optional[datetime] = None

    def analyze_context(self, messages: List[Dict[str, str]]) -> str:
        """Analyze contextual information from conversation history."""
        if not messages:
            return "No prior context available."
        
        recent_topics = []
        for msg in messages[-5:]:  # Last 5 messages
            content = msg.get("content", "")
            if len(content) > 20:
                recent_topics.append(content[:50] + "...")
        
        return f"Recent conversation topics: {', '.join(recent_topics) if recent_topics else 'None'}"

    def analyze_intent(self, user_input: str) -> str:
        """Analyze user intent from input patterns."""
        intent_keywords = {
            "question": ["what", "how", "why", "when", "where", "?"],
            "command": ["create", "generate", "make", "build", "do"],
            "analysis": ["analyze", "explain", "describe", "summarize"],
            "comparison": ["compare", "difference", "versus", "vs"],
        }
        
        user_lower = user_input.lower()
        detected_intents = []
        for intent, keywords in intent_keywords.items():
            if any(kw in user_lower for kw in keywords):
                detected_intents.append(intent)
                self.user_intent_patterns[intent] = self.user_intent_patterns.get(intent, 0) + 1
        
        return f"Detected intent: {', '.join(detected_intents) if detected_intents else 'general inquiry'}"

    def analyze_behavior(self, user_input: str, history: List[Dict[str, str]]) -> str:
        """Analyze user behavior patterns."""
        input_length = len(user_input)
        avg_history_length = (
            sum(len(m.get("content", "")) for m in history) / len(history)
            if history else 0
        )
        
        behavior_type = "detailed" if input_length > 100 else "concise"
        engagement_level = "high" if len(history) > 3 else "low"
        
        self.behavior_metrics["avg_input_length"] = avg_history_length
        self.behavior_metrics["engagement_level"] = engagement_level
        
        return f"User behavior: {behavior_type} input style, {engagement_level} engagement level"

    def analyze_temporal(self) -> str:
        """Analyze temporal context."""
        now = datetime.now()
        self.temporal_context = now
        
        hour = now.hour
        time_of_day = (
            "morning" if 5 <= hour < 12
            else "afternoon" if 12 <= hour < 17
            else "evening" if 17 <= hour < 21
            else "night"
        )
        
        return f"Temporal context: {time_of_day} ({now.strftime('%Y-%m-%d %H:%M:%S')})"

    def generate_4d_analysis(self, user_input: str, messages: List[Dict[str, str]]) -> str:
        """Generate comprehensive 4D analysis."""
        context = self.analyze_context(messages)
        intent = self.analyze_intent(user_input)
        behavior = self.analyze_behavior(user_input, messages)
        temporal = self.analyze_temporal()
        
        return f"""4D Prompt Agent Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Context Dimension: {context}
🎯 Intent Dimension: {intent}
👤 Behavior Dimension: {behavior}
⏰ Temporal Dimension: {temporal}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


class GroqPromptingSystem:
    """Prompting system using Groq API with 4D prompt agent and RAG support."""

    def __init__(
        self,
        config_path: pathlib.Path,
        api_key: Optional[str] = None,
        rag_data_path: Optional[pathlib.Path] = None,
        station_data_path: Optional[pathlib.Path] = None,
    ):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Ensure paths are Path objects
        if isinstance(rag_data_path, str):
            rag_data_path = pathlib.Path(rag_data_path)
        if isinstance(station_data_path, str):
            station_data_path = pathlib.Path(station_data_path)
        # API key resolution order: CLI arg -> environment var -> config file
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY") or self.config.get("api_key")
        if not self.api_key:
            raise ValueError("Mistral API key required. Set via --api-key, MISTRAL_API_KEY environment variable, or in config file.")
        
        self.history: List[Dict[str, str]] = []
        self.four_d_agent = FourDPromptAgent()
        
        # Initialize RAG if data path provided
        self.rag_retriever = None
        if rag_data_path and rag_data_path.exists():
            try:
                from rag_system import RAGRetriever
                self.rag_retriever = RAGRetriever(rag_data_path)
                print(f"✓ RAG enabled: Loaded dataset from {rag_data_path}")
            except Exception as e:
                print(f"Warning: Could not load RAG system: {e}")

        # Initialize Station Retriever if data path provided
        self.station_retriever = None
        if station_data_path and station_data_path.exists():
            try:
                from rag_system import StationRetriever
                self.station_retriever = StationRetriever(station_data_path)
                print(f"✓ Station RAG enabled: Loaded dataset from {station_data_path}")
            except Exception as e:
                print(f"Warning: Could not load Station RAG system: {e}")

    def _load_config(self, path: pathlib.Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file {path} not found.")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @property
    def model(self) -> str:
        return self.config.get("model", "mistral-large-latest")

    def _build_system_prompt(self, user_input: Optional[str] = None) -> str:
        """Build system prompt with 4D agent analysis."""
        base_system = self.config.get(
            "system_prompt",
            "You are a helpful AI assistant with advanced multi-dimensional understanding."
        )
        
        # Add RAG-specific instruction if RAG is enabled
        rag_instruction = ""
        if self.rag_retriever:
            rag_instruction = "\n\n⚠️ CRITICAL RAG INSTRUCTION: The provided dataset contains EV charging session data. If the user asks for statistics (total, average, etc.), YOU MUST use the pre-calculated statistics provided in the context. However, if the user asks a general question or a question that cannot be answered by the dataset (e.g., general advice, definitions, or location questions where no station data is found), YOU SHOULD use your general knowledge. Do not force the dataset into the answer if it's not relevant."
        
        if self.station_retriever:
            rag_instruction += "\n\n📍 LOCATION INSTRUCTION: When providing charging station recommendations, you MUST provide the following details for EACH station:\n1. Station Name\n2. Full Address\n3. Distance\n4. The 'Map: [View on Google Maps](...)' link provided in the context.\n\nDo not summarize or omit the address or map link."
        
        # Add 4D analysis if we have user input and history
        if user_input and self.history:
            four_d_analysis = self.four_d_agent.generate_4d_analysis(user_input, self.history)
            return f"""{base_system}{rag_instruction}

{four_d_analysis}

Use the 4D analysis above to provide contextually aware, intent-aligned, behavior-adapted, and temporally relevant responses. If the dataset doesn't contain the answer, say so and provide a helpful general response."""
        
        return base_system + rag_instruction

    def _prepare_messages(
        self,
        user_input: str,
        include_4d: bool = True,
        use_rag: bool = True,
        top_k: int = 5,
    ) -> List[Dict[str, str]]:
        """Prepare messages for Groq API with optional RAG."""
        messages = []
        
        # System prompt with 4D analysis
        system_content = self._build_system_prompt(user_input if include_4d else None)
        messages.append({"role": "system", "content": system_content})
        
        # Add RAG context if available and enabled
        rag_context = ""
        if use_rag and self.rag_retriever:
            user_lower = user_input.lower()
            
            # Detect if this is an aggregate/statistical question
            aggregate_keywords = [
                "total", "all", "average", "mean", "sum", "count", "how many",
                "statistics", "statistic", "overall", "entire", "complete",
                "maximum", "minimum", "max", "min", "range", "distribution",
                "calculate", "calculation", "divide", "divided", "dividing",
                "multiply", "multiplied", "multiplying", "kwh", "energy",
                "sessions", "session", "charging time", "time", "duration",
                "hours", "minutes", "how long", "timezone", "time zone",
                "popular", "common", "most", "trend", "distribution",
                "averge", "avrage", "total", "sum"
            ]
            is_aggregate_query = any(kw in user_lower for kw in aggregate_keywords)
            
            if is_aggregate_query:
                # For aggregate questions, calculate comprehensive stats from ALL sessions
                all_stats = self.rag_retriever.get_all_sessions_stats()
                
                # Build comprehensive context with ALL data statistics
                total_sessions = all_stats.get('total_sessions', 0)
                total_kwh = all_stats.get('total_kwh', 0)
                avg_kwh = all_stats.get('average_kwh', 0)
                min_kwh = all_stats.get('min_kwh', 0)
                max_kwh = all_stats.get('max_kwh', 0)
                unique_stations = all_stats.get('unique_stations', 0)
                unique_sites = all_stats.get('unique_sites', 0)
                unique_clusters = all_stats.get('unique_clusters', 0)
                avg_charging_time = all_stats.get('average_charging_time_hours', 0)
                min_charging_time = all_stats.get('min_charging_time_hours', 0)
                max_charging_time = all_stats.get('max_charging_time_hours', 0)
                max_charging_time = all_stats.get('max_charging_time_hours', 0)
                sessions_with_charging_time = all_stats.get('sessions_with_charging_time', 0)
                timezones = all_stats.get('timezones', {})
                most_common_timezone = all_stats.get('most_common_timezone', 'Unknown')
                
                # Calculate verification: show the actual calculation
                calculated_avg = total_kwh / total_sessions if total_sessions > 0 else 0
                
                rag_context = f"""
═══════════════════════════════════════════════════════════════
DATASET: COMPLETE STATISTICS FROM ALL {total_sessions:,} SESSIONS
═══════════════════════════════════════════════════════════════

⚠️⚠️⚠️ CRITICAL: NO EXAMPLE SESSIONS SHOWN - USE ONLY THE STATS BELOW ⚠️⚠️⚠️

These statistics are PRE-CALCULATED from ALL {total_sessions:,} charging sessions.
DO NOT look for example sessions - they are NOT provided for aggregate queries.
DO NOT calculate from any subset - use ONLY the numbers below.

═══════════════════════════════════════════════════════════════
PRE-CALCULATED STATISTICS (FROM ALL {total_sessions:,} SESSIONS):
═══════════════════════════════════════════════════════════════

Total Sessions: {total_sessions:,}
Total Energy Delivered: {total_kwh:,.2f} kWh
Average Energy per Session: {avg_kwh:.2f} kWh
Minimum Energy per Session: {min_kwh:.2f} kWh
Maximum Energy per Session: {max_kwh:.2f} kWh
Unique Stations: {unique_stations}
        Maximum Charging Time: {max_charging_time:.2f} hours ({max_charging_time * 60:.1f} minutes)

                Timezone Statistics:
                Most Common Timezone: {most_common_timezone}
                Timezone Distribution: {json.dumps(timezones, indent=2)}

VERIFICATION CALCULATION:
- Total Energy ({total_kwh:,.2f} kWh) ÷ Total Sessions ({total_sessions:,}) = Average ({avg_kwh:.2f} kWh)
- This calculation uses ALL {total_sessions:,} sessions, NOT 5 sessions

═══════════════════════════════════════════════════════════════
⚠️⚠️⚠️ MANDATORY: USE THESE EXACT NUMBERS - DO NOT RECALCULATE ⚠️⚠️⚠️
═══════════════════════════════════════════════════════════════

FOR ANY QUESTION ABOUT:
- Total energy → Answer: {total_kwh:,.2f} kWh (from {total_sessions:,} sessions)
- Average energy → Answer: {avg_kwh:.2f} kWh per session (from {total_sessions:,} sessions)
- Average charging time → Answer: {avg_charging_time:.2f} hours ({avg_charging_time * 60:.1f} minutes) from {sessions_with_charging_time:,} sessions
- Number of sessions → Answer: {total_sessions:,} sessions
- Any calculation → Use the pre-calculated numbers above

FORBIDDEN ACTIONS:
❌ DO NOT divide {total_kwh:,.2f} by 5
❌ DO NOT use 38.50 kWh (that's from only 5 sessions - WRONG!)
❌ DO NOT use 7.70 kWh (that's from only 5 sessions - WRONG!)
❌ DO NOT look for example sessions (they are NOT shown for aggregate queries)
❌ DO NOT calculate from any subset of sessions

CORRECT CALCULATION:
✅ Total Energy ({total_kwh:,.2f} kWh) ÷ Sessions ({total_sessions:,}) = {avg_kwh:.2f} kWh per session
✅ This is the ONLY correct calculation - it uses ALL {total_sessions:,} sessions

═══════════════════════════════════════════════════════════════
FINAL ANSWER TEMPLATE:
═══════════════════════════════════════════════════════════════

If asked about average energy: "The average energy delivered per session is {avg_kwh:.2f} kWh, calculated from all {total_sessions:,} sessions in the dataset (total: {total_kwh:,.2f} kWh)."

If asked about average charging time: "The average charging time is {avg_charging_time:.2f} hours ({avg_charging_time * 60:.1f} minutes), calculated from {sessions_with_charging_time:,} sessions with complete time data."

If asked about total: "The total energy delivered across all sessions is {total_kwh:,.2f} kWh, from {total_sessions:,} charging sessions."

If asked to calculate: "Dividing the total energy ({total_kwh:,.2f} kWh) by the number of sessions ({total_sessions:,}) gives an average of {avg_kwh:.2f} kWh per session."

═══════════════════════════════════════════════════════════════

"""
            else:
                # Check for location-based queries if Station Retriever is enabled
                location_keywords = ["nearest", "closest", "near", "nearby", "location", "station", "map"]
                is_location_query = any(kw in user_lower for kw in location_keywords)
                
                if self.station_retriever and is_location_query:
                    # Attempt to geocode the query
                    location_data = self._geocode_location(user_input, self.history)
                    
                    # Fallback to user location if no specific location found but "nearest" is asked
                    if not location_data and any(kw in user_lower for kw in ["nearest", "closest", "near", "nearby"]):
                        print("No specific location found, detecting user location...")
                        location_data = self._get_user_location()
                    
                    if location_data:
                        lat = location_data["latitude"]
                        lon = location_data["longitude"]
                        loc_name = location_data.get("location_name", "target location")
                        
                        nearest_stations = self.station_retriever.find_nearest(lat, lon, limit=top_k)
                        station_context = self.station_retriever.format_station_context(nearest_stations)
                        
                        rag_context = f"""
═══════════════════════════════════════════════════════════════
LOCATION SEARCH RESULTS
═══════════════════════════════════════════════════════════════
Reference Location: {loc_name} ({lat}, {lon})

{station_context}

═══════════════════════════════════════════════════════════════
"""
                    else:
                         # If it was a location query but we couldn't find a location, 
                         # DO NOT fallback to session data. It confuses the user.
                         # Instead, provide a hint in the context.
                         rag_context = "\n\n[SYSTEM NOTE: The user asked for a location/station, but no specific location could be determined from the query, history, or current IP. Please ask the user to specify a location (e.g., 'near Hollywood'). Do NOT use the session dataset.]\n\n"

                else:
                    # For specific queries, retrieve relevant sessions
                    relevant_sessions = self.rag_retriever.retrieve(user_input, top_k=top_k)
                    if relevant_sessions:
                        rag_context = f"\n\nRelevant data from dataset ({len(relevant_sessions)} sessions found):\n{self.rag_retriever.format_context(relevant_sessions, max_sessions=top_k, include_all_stats=False)}\n\n"
        
        # Conversation history (excluding system messages)
        for msg in self.history:
            if msg.get("role") != "system":
                messages.append(msg)
        
        # Current user input with RAG context
        enhanced_input = f"{rag_context}Question: {user_input.strip()}\n\nPlease answer based on the provided data if available, otherwise use your general knowledge."
        messages.append({"role": "user", "content": enhanced_input})
        
        return messages

    def _call_groq(self, messages: List[Dict[str, str]]) -> str:
        """Call Groq API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
        }
        
        # Add optional parameters from config
        if "temperature" in self.config:
            payload["temperature"] = self.config["temperature"]
        if "max_tokens" in self.config:
            payload["max_tokens"] = self.config["max_tokens"]
        
        response = requests.post(MISTRAL_API_URL, json=payload, headers=headers, timeout=120)
        
        # Handle payment/credit errors gracefully
        if response.status_code == 402:
            error_msg = response.json().get("error", {}).get("message", "Payment required")
            return f"⚠️ Mistral API Error: {error_msg}\n\nPlease check your Mistral account.\n\nHowever, I can provide you with the dataset statistics:\n"
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            print(f"Request payload: {json.dumps(payload, indent=2)}")
            response.raise_for_status()
        
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        
        raise ValueError("Unexpected Mistral response format.")

    def _get_user_location(self) -> Optional[Dict[str, float]]:
        """Get user location based on IP address."""
        try:
            response = requests.get("http://ip-api.com/json/", timeout=5)
            data = response.json()
            if data.get("status") == "success":
                return {
                    "latitude": data.get("lat"),
                    "longitude": data.get("lon"),
                    "location_name": f"Current Location ({data.get('city')}, {data.get('regionName')})"
                }
        except Exception as e:
            print(f"Location detection error: {e}")
        return None

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
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        try:
            response = self._call_groq(messages)
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

    def ask(
        self,
        user_input: str,
        include_4d: bool = True,
        use_rag: bool = True,
        top_k: int = 5,
    ) -> str:
        """Ask a question with 4D prompt agent analysis and optional RAG."""
        messages = self._prepare_messages(user_input, include_4d, use_rag, top_k)
        answer = self._call_groq(messages)
        
        # Update history (store original user input, not enhanced)
        self.history.append({"role": "user", "content": user_input.strip()})
        self.history.append({"role": "assistant", "content": answer})
        
        # Keep history manageable (last 20 messages)
        if len(self.history) > 20:
            self.history = self.history[-20:]
        
        return answer

    def save_transcript(self, out_dir: pathlib.Path) -> pathlib.Path:
        """Save conversation transcript."""
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"conversation-{timestamp}.json"
        
        transcript = {
            "timestamp": timestamp,
            "model": self.model,
            "history": self.history,
            "4d_metrics": {
                "intent_patterns": self.four_d_agent.user_intent_patterns,
                "behavior_metrics": self.four_d_agent.behavior_metrics,
            },
        }
        
        out_path.write_text(
            json.dumps(transcript, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mistral prompting system with 4D prompt agent."
    )
    parser.add_argument(
        "--config",
        default="/Users/omaryasser/CascadeProjects/openrouter_config.json",
        type=pathlib.Path,
        help="Path to config JSON file.",
    )
    parser.add_argument(
        "--api-key",
        help="Mistral API key (overrides config).",
    )
    parser.add_argument(
        "--no-4d",
        action="store_true",
        help="Disable 4D prompt agent analysis.",
    )
    parser.add_argument(
        "--rag-data",
        type=pathlib.Path,
        default="/Users/omaryasser/CascadeProjects/acndata_sessions_fixed.json",
        help="Path to JSON dataset for RAG (default: acndata_sessions_fixed.json).",
    )
    parser.add_argument(
        "--station-data",
        type=str,
        default="/Users/omaryasser/CascadeProjects/converted_detailed_stations.json",
        help="Path to station JSON file",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant items to retrieve for RAG (default: 5).",
    )
    parser.add_argument(
        "--save-transcript",
        type=pathlib.Path,
        help="Directory to save conversation transcripts.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    prompt_parser = subparsers.add_parser("prompt", help="Single-shot prompt.")
    prompt_parser.add_argument(
        "--input",
        required=True,
        help="User prompt to send.",
    )

    subparsers.add_parser("chat", help="Interactive chat loop.")

    return parser


def run_prompt_flow(system: GroqPromptingSystem, args: argparse.Namespace) -> None:
    answer = system.ask(
        args.input,
        include_4d=not args.no_4d,
        use_rag=not args.no_rag,
        top_k=args.top_k,
    )
    print(answer)


def run_chat_loop(system: GroqPromptingSystem, args: argparse.Namespace) -> None:
    rag_status = "enabled" if system.rag_retriever and not args.no_rag else "disabled"
    print(f"Entering interactive chat with 4D prompt agent (RAG: {rag_status}). Type /exit to quit.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break
        
        if user_input.lower() in {"/exit", "/quit"}:
            break
        
        if not user_input:
            continue
        
        answer = system.ask(
            user_input,
            include_4d=not args.no_4d,
            use_rag=not args.no_rag,
            top_k=args.top_k,
        )
        print(f"\nassistant> {answer}\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        system = GroqPromptingSystem(
            args.config,
            api_key=args.api_key,
            rag_data_path=args.rag_data if not args.no_rag else None,
            station_data_path=args.station_data if not args.no_rag else None,
        )
    except Exception as e:
        print(f"Error initializing system: {e}", file=sys.stderr)
        return 1

    if args.command == "prompt":
        run_prompt_flow(system, args)
    elif args.command == "chat":
        run_chat_loop(system, args)
    else:
        parser.error("Unsupported command.")

    if args.save_transcript:
        path = system.save_transcript(args.save_transcript)
        print(f"\nTranscript saved to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

