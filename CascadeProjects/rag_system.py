#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) system for acndata_sessions_fixed.json.

Features:
  * Loads and indexes EV charging session data
  * Performs semantic/keyword search on the dataset
  * Retrieves relevant context for questions
  * Integrates with OpenRouter prompting system
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional
from datetime import datetime
import re


class RAGRetriever:
    """Retrieval system for EV charging session data."""

    def __init__(self, data_path: pathlib.Path):
        self.data_path = data_path
        self.sessions: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load JSON data from file."""
        print(f"Loading data from {self.data_path}...", end=" ", flush=True)
        with self.data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.metadata = data.get("_meta", {})
        self.sessions = data.get("_items", [])
        print(f"Loaded {len(self.sessions)} charging sessions.")

    def _session_to_text(self, session: Dict[str, Any]) -> str:
        """Convert a session to searchable text."""
        parts = []
        if session.get("sessionID"):
            parts.append(f"Session ID: {session['sessionID']}")
        if session.get("stationID"):
            parts.append(f"Station: {session['stationID']}")
        if session.get("spaceID"):
            parts.append(f"Space: {session['spaceID']}")
        if session.get("siteID"):
            parts.append(f"Site: {session['siteID']}")
        if session.get("clusterID"):
            parts.append(f"Cluster: {session['clusterID']}")
        if session.get("kWhDelivered") is not None:
            parts.append(f"Energy delivered: {session['kWhDelivered']} kWh")
        if session.get("connectionTime"):
            parts.append(f"Connected: {session['connectionTime']}")
        if session.get("disconnectTime"):
            parts.append(f"Disconnected: {session['disconnectTime']}")
        if session.get("doneChargingTime"):
            parts.append(f"Charging completed: {session['doneChargingTime']}")
        if session.get("timezone"):
            parts.append(f"Timezone: {session['timezone']}")
        
        return " | ".join(parts)

    def _calculate_relevance_score(
        self, session: Dict[str, Any], query: str
    ) -> float:
        """Calculate relevance score for a session based on query."""
        query_lower = query.lower()
        session_text = self._session_to_text(session).lower()
        
        score = 0.0
        
        # Exact keyword matches
        keywords = query_lower.split()
        for keyword in keywords:
            if keyword in session_text:
                score += 1.0
        
        # Specific field matches (higher weight)
        if "station" in query_lower and session.get("stationID"):
            if any(kw in str(session["stationID"]).lower() for kw in keywords):
                score += 3.0
        if "site" in query_lower and session.get("siteID"):
            if any(kw in str(session["siteID"]).lower() for kw in keywords):
                score += 3.0
        if "cluster" in query_lower and session.get("clusterID"):
            if any(kw in str(session["clusterID"]).lower() for kw in keywords):
                score += 3.0
        if "space" in query_lower and session.get("spaceID"):
            if any(kw in str(session["spaceID"]).lower() for kw in keywords):
                score += 3.0
        
        # Date/time matches
        if any(word in query_lower for word in ["date", "time", "day", "month", "year", "2018", "2019", "2020"]):
            if session.get("connectionTime"):
                if any(kw in session["connectionTime"].lower() for kw in keywords if len(kw) > 3):
                    score += 2.0
        
        # Energy/kWh matches
        if any(word in query_lower for word in ["kwh", "energy", "delivered", "charge"]):
            if session.get("kWhDelivered") is not None:
                score += 2.0
        
        return score

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant sessions for a query."""
        if not self.sessions:
            return []
        
        # Calculate relevance scores
        scored_sessions = []
        for session in self.sessions:
            score = self._calculate_relevance_score(session, query)
            if score > 0:
                scored_sessions.append((score, session))
        
        # Sort by score (descending) and return top-k
        scored_sessions.sort(key=lambda x: x[0], reverse=True)
        return [session for _, session in scored_sessions[:top_k]]

    def format_context(
        self, sessions: List[Dict[str, Any]], max_sessions: int = 5, include_all_stats: bool = False
    ) -> str:
        """Format retrieved sessions as context text."""
        if not sessions:
            return "No relevant charging sessions found in the dataset."
        
        context_parts = []
        
        # Add overall dataset statistics if requested (for aggregate questions)
        if include_all_stats:
            all_stats = self.get_dataset_stats()
            context_parts.append(all_stats)
            context_parts.append("\n" + "=" * 60 + "\n")
        
        context_parts.append(f"Relevant EV charging session data ({len(sessions)} sessions):")
        context_parts.append("=" * 60)
        
        for i, session in enumerate(sessions[:max_sessions], 1):
            context_parts.append(f"\nSession {i}:")
            context_parts.append(self._session_to_text(session))
        
        # Add summary statistics for retrieved sessions
        if len(sessions) > 1:
            total_kwh = sum(
                s.get("kWhDelivered", 0) for s in sessions if s.get("kWhDelivered")
            )
            avg_kwh = total_kwh / len(sessions) if sessions else 0
            context_parts.append(f"\n\nRetrieved Sessions Summary: {len(sessions)} sessions, "
                               f"Total kWh: {total_kwh:.2f}, "
                               f"Average kWh: {avg_kwh:.2f}")
        
        return "\n".join(context_parts)
    
    def get_all_sessions_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics from all sessions."""
        if not self.sessions:
            return {}
        
        total_sessions = len(self.sessions)
        sessions_with_kwh = [s for s in self.sessions if s.get("kWhDelivered") is not None]
        
        total_kwh = sum(s.get("kWhDelivered", 0) for s in sessions_with_kwh)
        avg_kwh = total_kwh / len(sessions_with_kwh) if sessions_with_kwh else 0
        
        kwh_values = [s.get("kWhDelivered", 0) for s in sessions_with_kwh if s.get("kWhDelivered") is not None]
        min_kwh = min(kwh_values) if kwh_values else 0
        max_kwh = max(kwh_values) if kwh_values else 0
        
        sites = set(s.get("siteID") for s in self.sessions if s.get("siteID"))
        stations = set(s.get("stationID") for s in self.sessions if s.get("stationID"))
        clusters = set(s.get("clusterID") for s in self.sessions if s.get("clusterID"))
        
        # Calculate charging time statistics
        charging_times_seconds = []
        for session in self.sessions:
            connection_time = session.get("connectionTime")
            done_charging_time = session.get("doneChargingTime")
            if connection_time and done_charging_time:
                try:
                    # Parse dates (format: "Wed, 25 Apr 2018 11:08:04 GMT")
                    from datetime import datetime
                    conn_dt = datetime.strptime(connection_time, "%a, %d %b %Y %H:%M:%S %Z")
                    done_dt = datetime.strptime(done_charging_time, "%a, %d %b %Y %H:%M:%S %Z")
                    charging_time_seconds = (done_dt - conn_dt).total_seconds()
                    if charging_time_seconds > 0:
                        charging_times_seconds.append(charging_time_seconds)
                except Exception:
                    pass
        
        avg_charging_time_hours = 0
        min_charging_time_hours = 0
        max_charging_time_hours = 0
        if charging_times_seconds:
            avg_charging_time_hours = sum(charging_times_seconds) / len(charging_times_seconds) / 3600
            min_charging_time_hours = min(charging_times_seconds) / 3600
            max_charging_time_hours = max(charging_times_seconds) / 3600
            
        # Timezone statistics
        timezones = [s.get("timezone") for s in self.sessions if s.get("timezone")]
        timezone_counts = {}
        for tz in timezones:
            timezone_counts[tz] = timezone_counts.get(tz, 0) + 1
        
        most_common_timezone = None
        if timezone_counts:
            most_common_timezone = max(timezone_counts.items(), key=lambda x: x[1])[0]
        
        return {
            "total_sessions": total_sessions,
            "sessions_with_kwh": len(sessions_with_kwh),
            "total_kwh": total_kwh,
            "average_kwh": avg_kwh,
            "min_kwh": min_kwh,
            "max_kwh": max_kwh,
            "unique_sites": len(sites),
            "unique_stations": len(stations),
            "unique_clusters": len(clusters),
            "sessions_with_charging_time": len(charging_times_seconds),
            "average_charging_time_hours": avg_charging_time_hours,
            "min_charging_time_hours": min_charging_time_hours,
            "max_charging_time_hours": max_charging_time_hours,
            "timezones": timezone_counts,
            "most_common_timezone": most_common_timezone,
        }

    def get_dataset_stats(self) -> str:
        """Get overall dataset statistics."""
        if not self.sessions:
            return "No data available."
        
        total_sessions = len(self.sessions)
        total_kwh = sum(
            s.get("kWhDelivered", 0) for s in self.sessions
            if s.get("kWhDelivered") is not None
        )
        avg_kwh = total_kwh / total_sessions if total_sessions > 0 else 0
        
        sites = set(s.get("siteID") for s in self.sessions if s.get("siteID"))
        stations = set(s.get("stationID") for s in self.sessions if s.get("stationID"))
        
        stats = [
            f"Dataset Statistics:",
            f"- Total sessions: {total_sessions:,}",
            f"- Total energy delivered: {total_kwh:,.2f} kWh",
            f"- Average energy per session: {avg_kwh:.2f} kWh",
            f"- Unique sites: {len(sites)}",
            f"- Unique stations: {len(stations)}",
        ]
        
        if self.metadata:
            if self.metadata.get("site"):
                stats.append(f"- Site: {self.metadata['site']}")
            if self.metadata.get("min_kWh") is not None:
                stats.append(f"- Min kWh: {self.metadata['min_kWh']}")
        
        return "\n".join(stats)


class RAGPromptingSystem:
    """RAG-enhanced prompting system with OpenRouter."""

    def __init__(
        self,
        config_path: pathlib.Path,
        data_path: pathlib.Path,
        api_key: Optional[str] = None,
    ):
        from openrouter_prompting_system import OpenRouterPromptingSystem, FourDPromptAgent
        
        self.rag_retriever = RAGRetriever(data_path)
        self.base_system = OpenRouterPromptingSystem(config_path, api_key)
        self.four_d_agent = FourDPromptAgent()

    def ask(
        self,
        user_input: str,
        top_k: int = 5,
        include_4d: bool = True,
        local: bool = False,
    ) -> str:
        """Ask a question with RAG retrieval."""
        # Retrieve relevant context
        relevant_sessions = self.rag_retriever.retrieve(user_input, top_k=top_k)
        context = self.rag_retriever.format_context(relevant_sessions, max_sessions=top_k)
        
        # Enhance user input with context
        enhanced_prompt = f"""Based on the following EV charging session data, answer the question:

{context}

Question: {user_input}

Provide a detailed answer based on the data above. If the data doesn't contain enough information, say so."""
        
        # If running in local/offline mode, generate an offline answer using RAG only
        if local:
            # Handle aggregate or statistical queries by returning global dataset stats
            q_lower = user_input.lower()
            aggregate_keywords = [
                "total", "all", "average", "mean", "sum", "count", "how many",
                "statistics", "statistic", "overall", "entire", "complete",
                "maximum", "minimum", "max", "min", "range", "distribution",
                "calculate", "calculation", "kwh", "energy",
                "sessions", "session", "charging time", "time", "duration",
                "hours", "minutes", "how long"
            ]
            if any(kw in q_lower for kw in aggregate_keywords):
                # Use the comprehensive stats calculated from all sessions
                stats = self.rag_retriever.get_all_sessions_stats()
                if not stats:
                    return "No dataset statistics available."

                lines = [
                    "Dataset (offline) — aggregated statistics:",
                    f"- Total sessions: {stats.get('total_sessions', 0):,}",
                    f"- Total energy delivered: {stats.get('total_kwh', 0):,.2f} kWh",
                    f"- Average energy per session: {stats.get('average_kwh', 0):.2f} kWh",
                    f"- Min energy per session: {stats.get('min_kwh', 0):.2f} kWh",
                    f"- Max energy per session: {stats.get('max_kwh', 0):.2f} kWh",
                    f"- Unique sites: {stats.get('unique_sites', 0)}",
                    f"- Unique stations: {stats.get('unique_stations', 0)}",
                    f"- Unique clusters: {stats.get('unique_clusters', 0)}",
                ]
                if stats.get('sessions_with_charging_time'):
                    lines.extend([
                        f"- Sessions with charging time data: {stats.get('sessions_with_charging_time')}",
                        f"- Average charging time: {stats.get('average_charging_time_hours', 0):.2f} hours",
                    ])

                return "\n".join(lines)

            # For non-aggregate queries, return retrieved context and a short summary
            if relevant_sessions:
                return self.rag_retriever.format_context(relevant_sessions, max_sessions=top_k, include_all_stats=False)

            return "No relevant charging sessions found in the dataset (offline search)."

        # Use the base system to get the answer (online/OpenRouter)
        return self.base_system.ask(enhanced_prompt, include_4d=include_4d)

    def get_stats(self) -> str:
        """Get dataset statistics."""
        return self.rag_retriever.get_dataset_stats()



class StationRetriever:
    """Retrieval system for LA EV charging stations."""

    def __init__(self, data_path: pathlib.Path):
        self.data_path = data_path
        self.stations: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load JSON data from file."""
        if not self.data_path.exists():
            print(f"Warning: Station data file not found at {self.data_path}")
            return

        print(f"Loading station data from {self.data_path}...", end=" ", flush=True)
        try:
            with self.data_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.stations = data.get("fuel_stations", [])
            print(f"Loaded {len(self.stations)} stations.")
        except Exception as e:
            print(f"Error loading station data: {e}")

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance between two points in km."""
        import math
        R = 6371  # Earth radius in km

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) * math.sin(dlon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def find_nearest(self, lat: float, lon: float, limit: int = 5) -> List[Dict[str, Any]]:
        """Find nearest stations to a coordinate."""
        if not self.stations:
            return []

        scored_stations = []
        for station in self.stations:
            try:
                s_lat = station.get("latitude")
                s_lon = station.get("longitude")
                
                if s_lat is not None and s_lon is not None:
                    dist = self._haversine_distance(lat, lon, float(s_lat), float(s_lon))
                    scored_stations.append((dist, station))
            except (ValueError, TypeError):
                continue

        # Sort by distance (ascending)
        scored_stations.sort(key=lambda x: x[0])
        
        # Return top-k with distance info added
        results = []
        for dist, station in scored_stations[:limit]:
            station_copy = station.copy()
            station_copy["_distance_km"] = dist
            results.append(station_copy)
            
        return results

    def format_station_context(self, stations: List[Dict[str, Any]]) -> str:
        """Format station data as context text."""
        if not stations:
            return "No nearby charging stations found."

        parts = ["Nearby Charging Stations:"]
        for i, s in enumerate(stations, 1):
            dist = s.get("_distance_km", 0)
            name = s.get("station_name", "Unknown Station")
            address = f"{s.get('street_address')}, {s.get('city')}, {s.get('state')} {s.get('zip')}"
            phone = s.get("station_phone")
            access = s.get("access_days_time")
            network = s.get("ev_network")
            
            # Format connector info
            connectors = []
            if s.get("ev_connector_types"):
                connectors = s.get("ev_connector_types")
            
            # Generate Google Maps link
            lat = s.get("latitude")
            lon = s.get("longitude")
            map_link = ""
            if lat is not None and lon is not None:
                map_link = f"[View on Google Maps](https://www.google.com/maps/search/?api=1&query={lat},{lon})"
            
            parts.append(f"\n{i}. {name} ({dist:.2f} km away)")
            parts.append(f"   Address: {address}")
            if map_link:
                parts.append(f"   Map: {map_link}")
            if phone:
                parts.append(f"   Phone: {phone}")
            if access:
                parts.append(f"   Access: {access}")
            if network:
                parts.append(f"   Network: {network}")
            
            # New fields from detailed dataset
            pricing = s.get("ev_pricing")
            if pricing:
                parts.append(f"   Pricing: {pricing}")
            
            rating = s.get("rating")
            if rating:
                parts.append(f"   Rating: {rating}/5.0")
            
            charger_type = s.get("charger_type")
            if charger_type:
                parts.append(f"   Type: {charger_type}")
                
            usage = s.get("usage_stats")
            if usage:
                parts.append(f"   Avg Usage: {usage} users/day")
            
            if connectors:
                parts.append(f"   Connectors: {', '.join(connectors)}")
                
        return "\n".join(parts)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG system for EV charging data")
    parser.add_argument(
        "--config",
        default="/Users/omaryasser/CascadeProjects/openrouter_config.json",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--data",
        default="/Users/omaryasser/CascadeProjects/acndata_sessions_fixed.json",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--api-key",
        help="OpenRouter API key",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Question about the dataset",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant sessions to retrieve (default: 5)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics",
    )
    
    args = parser.parse_args()
    
    system = RAGPromptingSystem(args.config, args.data, api_key=args.api_key)
    
    if args.stats:
        print(system.get_stats())
        print("\n" + "=" * 60 + "\n")
    
    answer = system.ask(args.query, top_k=args.top_k)
    print(answer)

