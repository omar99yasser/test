# CascadeProjects - EV Route Planning & Charging Station Intelligence System

A comprehensive electric vehicle (EV) route planning and charging station intelligence platform, specifically designed for BMW i3 vehicles. This project combines route planning algorithms, AI-powered natural language processing, and retrieval-augmented generation (RAG) to provide intelligent EV journey planning.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Components](#project-components)
- [Features](#features)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Data Files](#data-files)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project is a multi-component system that helps EV drivers plan routes, find charging stations, and get intelligent recommendations based on:
- Current battery level
- Driving style (acceleration/braking patterns)
- Weather conditions
- Route distance and waypoints
- Real-time charging station availability and pricing

The system uses AI (Mistral/Groq/DeepSeek) for natural language query processing and RAG (Retrieval-Augmented Generation) for querying historical charging session data.

## 🧩 Project Components

### 1. **Route Planner Backend** (`route_planner_backend.py`)
Flask-based REST API server that handles route planning calculations and charging station recommendations.

**Key Features:**
- Calculates route distances using Haversine formula
- Adjusts energy consumption based on driving style and weather
- Finds nearby charging stations along routes
- Generates detailed charging plans with time and cost estimates
- Integrates with Mistral AI for natural language query processing
- Supports waypoints for multi-stop journeys

**Endpoints:**
- `POST /api/plan-route` - Calculate route plan with charging recommendations
- `GET /api/vehicle-specs` - Get BMW i3 specifications
- `POST /api/ask-llm` - Process natural language queries about trips

### 2. **Web Interface** (`ev_route_planner.html`)
Modern, responsive web interface for the route planner with:
- Interactive dashboard with sliders and gauges
- AI assistant for natural language trip planning
- Real-time route visualization
- Charging station recommendations with Google Maps integration
- Beautiful BMW-themed UI design

### 3. **RAG System** (`rag_system.py`)
Retrieval-Augmented Generation system for querying EV charging session data.

**Components:**
- **RAGRetriever**: Searches through charging session data (`acndata_sessions_fixed.json`)
- **RAGPromptingSystem**: Integrates RAG with OpenRouter/Mistral for intelligent responses
- **StationRetriever**: Finds nearby charging stations based on coordinates

**Features:**
- Semantic and keyword-based search
- Relevance scoring for session matching
- Aggregate statistics calculation (total kWh, averages, etc.)
- Context formatting for LLM prompts

### 4. **OpenRouter Prompting System** (`openrouter_prompting_system.py`)
Advanced LLM prompting system with 4D prompt agent framework.

**4D Dimensions:**
- **Context**: Conversation history and topics
- **Intent**: User goals and patterns (question, command, analysis, comparison)
- **Behavior**: User interaction style (detailed/concise, engagement level)
- **Temporal**: Time-based context (time of day, date)

**Features:**
- Supports multiple LLM providers (Mistral, Groq, DeepSeek)
- RAG integration for dataset queries
- Location-based station queries with geocoding
- Conversation history management
- Transcript saving

### 5. **Data Conversion Scripts**

**`convert_csv_to_json.py`**
- Converts CSV charging station data to JSON format
- Processes `detailed_ev_charging_stations.csv` → `converted_detailed_stations.json`
- Extracts station information, pricing, ratings, connector types

### 6. **Supporting Files**

**Setup Scripts:**
- `setup_mistral.sh` - Configures Mistral API environment
- `setup_deepseek.sh` - Configures DeepSeek API environment

**Test Files:**
- `test_backend.html` - Simple HTML test page for backend API
- `reproduce_issue.py` - Debugging script for geocoding issues

**Documentation:**
- `ROUTE_PLANNER_GUIDE.md` - Quick start guide for route planner
- `README_OPENROUTER.md` - OpenRouter system documentation
- `DEEPSEEK_SETUP.md` - DeepSeek API setup instructions
- `MISTRAL_SETUP.md` - Mistral API setup instructions

## ✨ Features

### Route Planning
- ✅ Multi-waypoint route support
- ✅ Real-time range calculation based on battery level
- ✅ Dynamic energy consumption adjustment:
  - Gas pedal level (acceleration style)
  - Brake pedal level (braking frequency)
  - Weather conditions (rain, snow, heat, cold, etc.)
- ✅ Automatic charging station discovery along routes
- ✅ Charging time and cost estimates
- ✅ Google Maps integration

### AI Assistant
- ✅ Natural language trip planning
- ✅ Extracts trip details from conversational queries:
  - Start/end locations
  - Waypoints
  - Battery level
  - Weather conditions
  - Driving style preferences
- ✅ Intelligent responses with context awareness

### Data Intelligence
- ✅ Query historical charging session data
- ✅ Aggregate statistics (total kWh, averages, etc.)
- ✅ Station-specific queries
- ✅ Location-based station search

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (HTML/JS)                   │
│              ev_route_planner.html                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Flask Backend (route_planner_backend.py)        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Route Calc   │  │ Station     │  │ Mistral AI   │      │
│  │ Engine       │  │ Retriever   │  │ Integration  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐  ┌──────────────────┐
│ RAG System       │  │ Station Data     │
│ (rag_system.py)  │  │ (JSON files)     │
└──────────────────┘  └──────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ OpenRouter/Mistral Prompting System     │
│ (openrouter_prompting_system.py)        │
│  - 4D Prompt Agent                      │
│  - RAG Integration                       │
│  - Location Geocoding                   │
└─────────────────────────────────────────┘
```

## 📦 Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### 1. Install Dependencies

```bash
pip install flask flask-cors requests
```

### 2. Configure API Keys

**Option A: Mistral AI (Currently Active)**
```bash
export MISTRAL_API_KEY="your-mistral-api-key"
# Or source the setup script:
source setup_mistral.sh
```

**Option B: DeepSeek**
```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
source setup_deepseek.sh
```

### 3. Prepare Data Files

Ensure you have the following data files:
- `bmwi3.json` - BMW i3 vehicle specifications
- `converted_detailed_stations.json` - Charging station data
- `acndata_sessions_fixed.json` - Historical charging session data (optional, for RAG)

If you need to convert CSV to JSON:
```bash
python3 convert_csv_to_json.py
```

### 4. Start the Backend Server

```bash
python3 route_planner_backend.py
```

The server will start on `http://localhost:5001`

### 5. Open the Web Interface

**Option A: Using Python HTTP Server**
```bash
python3 -m http.server 8000
# Then open: http://localhost:8000/ev_route_planner.html
```

**Option B: Direct File Access**
Open `ev_route_planner.html` directly in your browser (may have CORS limitations)

## 🚀 Usage

### Web Interface

1. **AI Assistant Mode:**
   - Enter a natural language query like:
     - "I'm going from Hollywood to Long Beach via Santa Monica with 30% battery in rainy weather"
     - "Driving aggressively from Pasadena to Santa Monica, it's snowing, 30% battery"
   - Click "Ask AI Assistant"
   - Get intelligent route plan with charging recommendations

2. **Dashboard Mode:**
   - Enter start and end locations
   - Adjust battery level using the gauge
   - Set acceleration level (gas pedal)
   - Set braking frequency
   - Select weather conditions
   - Click "Plan My Route"
   - View detailed results with charging stations

### Command Line (RAG System)

**Query charging session data:**
```bash
python3 rag_system.py \
  --config openrouter_config.json \
  --data acndata_sessions_fixed.json \
  --query "What is the average energy delivered per session?" \
  --top-k 5
```

**Interactive chat with RAG:**
```bash
python3 openrouter_prompting_system.py \
  --rag-data acndata_sessions_fixed.json \
  --station-data converted_detailed_stations.json \
  chat
```

**Single prompt:**
```bash
python3 openrouter_prompting_system.py \
  --rag-data acndata_sessions_fixed.json \
  prompt \
  --input "What is the total energy delivered across all sessions?"
```

### API Usage

**Plan a route:**
```bash
curl -X POST http://localhost:5001/api/plan-route \
  -H "Content-Type: application/json" \
  -d '{
    "start_location": "Hollywood",
    "end_location": "Long Beach",
    "battery_percent": 30,
    "gas_pedal_level": 70,
    "brake_pedal_level": 40,
    "weather_conditions": "rain"
  }'
```

**Ask AI about a trip:**
```bash
curl -X POST http://localhost:5001/api/ask-llm \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I need to go from Hollywood to Santa Monica with 50% battery in the rain"
  }'
```

## 📁 Data Files

### Vehicle Specifications
- **`bmwi3.json`**: BMW i3 eDrive35L specifications including:
  - Battery capacity (66.1 kWh usable)
  - Range (526 km)
  - Energy consumption (14.3 kWh/100km)
  - Charging capabilities (DC: 150 kW, AC: 11 kW)

### Charging Stations
- **`converted_detailed_stations.json`**: Detailed EV charging station data with:
  - Station names, addresses, coordinates
  - Pricing information
  - Ratings and reviews
  - Charger types and connector information
  - Network information
  - Usage statistics

- **`detailed_ev_charging_stations.csv`**: Source CSV file (can be converted to JSON)

- **`la_ev_charging_stations.json`**: Alternative LA charging station dataset

- **`evlocs.csv`**: Additional location data

### Charging Session Data
- **`acndata_sessions_fixed.json`**: Historical charging session data from Caltech ACN dataset:
  - Session IDs, station IDs, site IDs
  - Energy delivered (kWh)
  - Connection/disconnection times
  - Timezone information
  - Used for RAG queries and statistics

## 🔧 Configuration

### Backend Configuration (`route_planner_backend.py`)

**Mistral API Settings:**
```python
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "default-key")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
```

**Server Settings:**
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### OpenRouter Configuration (`openrouter_config.json`)

```json
{
  "api_key": "your-api-key",
  "model": "mistral-large-latest",
  "system_prompt": "Your custom system prompt...",
  "temperature": 0.7,
  "max_tokens": 2000
}
```

### RAG Configuration

**For aggregate queries:**
- Automatically calculates statistics from all sessions
- Provides pre-calculated totals, averages, min/max values

**For specific queries:**
- Retrieves top-k most relevant sessions
- Uses relevance scoring based on keywords and field matches

## 🐛 Troubleshooting

### Backend Not Starting
- Check if port 5001 is available: `lsof -i :5001`
- Verify Flask is installed: `pip install flask flask-cors`
- Check for import errors in `route_planner_backend.py`

### API Key Issues
- Verify API key is set: `echo $MISTRAL_API_KEY`
- Check API key in config file
- Ensure API key has sufficient credits/balance

### Station Data Not Loading
- Verify `converted_detailed_stations.json` exists
- Check file format (should have `fuel_stations` key)
- Run conversion script if needed: `python3 convert_csv_to_json.py`

### RAG Queries Not Working
- Ensure `acndata_sessions_fixed.json` exists and is valid JSON
- Check file size (should be ~65K+ lines)
- Verify RAG system can load data (check console output)

### CORS Errors in Browser
- Use Python HTTP server instead of opening file directly
- Or configure CORS in Flask backend (already enabled)

### Geocoding Issues
- Some locations may not be recognized (hardcoded list in `geocode_location()`)
- For production, integrate with Google Maps Geocoding API
- Check `reproduce_issue.py` for debugging geocoding

## 📊 Key Algorithms

### Energy Consumption Calculation

The system adjusts base consumption based on multiple factors:

```python
consumption = base_consumption * gas_multiplier * brake_multiplier * weather_multiplier
```

**Gas Pedal Multiplier:**
- 0% (eco mode): -20% consumption
- 50% (normal): baseline
- 100% (sport mode): +40% consumption

**Brake Pedal Multiplier:**
- 0%: baseline
- 100%: +15% consumption (less regenerative braking)

**Weather Multipliers:**
- Clear: 1.0x
- Cloudy: 1.05x
- Rain: 1.15x
- Heavy Rain: 1.25x
- Snow: 1.35x
- Windy: 1.10x
- Extreme Heat: 1.20x
- Cold: 1.25x

### Station Finding Algorithm

1. Divides route into segments
2. Samples multiple points along each segment
3. Finds nearest stations to each point
4. Filters stations within 15km of route
5. Calculates distance along route for each station
6. Sorts by distance from start
7. Recommends stations based on battery needs

### RAG Relevance Scoring

Scores sessions based on:
- Keyword matches in session text
- Field-specific matches (station ID, site ID, etc.) - 3x weight
- Date/time matches - 2x weight
- Energy/kWh matches - 2x weight

## 🔐 Security Notes

- API keys are stored in environment variables or config files
- Consider using `.gitignore` for config files with secrets
- For production, use proper secret management (AWS Secrets Manager, etc.)
- Google Maps API key in HTML should be restricted to specific domains

## 📝 License & Credits

- **ACN Data**: Caltech ACN (Adaptive Charging Network) dataset
- **BMW i3 Specs**: Based on official BMW i3 eDrive35L specifications
- **Charging Station Data**: Various public datasets

## 🚧 Future Enhancements

- [ ] Real-time charging station availability
- [ ] Integration with Google Maps Directions API for accurate routes
- [ ] Multi-vehicle support (not just BMW i3)
- [ ] User accounts and trip history
- [ ] Mobile app version
- [ ] Real-time weather API integration
- [ ] Charging network API integrations (ChargePoint, EVgo, etc.)
- [ ] Machine learning for consumption prediction

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the setup guides (MISTRAL_SETUP.md, DEEPSEEK_SETUP.md)
3. Check console logs for error messages
4. Verify all data files are present and valid

---

**Last Updated**: 2024
**Project Status**: Active Development


