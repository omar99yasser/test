from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import math
import os
import requests
from pathlib import Path
from rag_system import StationRetriever

app = Flask(__name__)
CORS(app)

# Mistral API configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "KYltzkacCJaxDwwPmVdjCN1qCBuclJbP")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Load BMW i3 specs
with open('bmwi3.json', 'r') as f:
    BMW_I3_SPECS = json.load(f)

# Initialize station retriever
station_retriever = StationRetriever(Path('converted_detailed_stations.json'))

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km"""
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def geocode_location(location_name):
    """Simple geocoding - in production, use Google Maps API"""
    # For demo, return some known locations
    locations = {
        'hollywood': (34.0928, -118.3287),
        'downtown la': (34.0522, -118.2437),
        'santa monica': (34.0195, -118.4912),
        'pasadena': (34.1478, -118.1445),
        'long beach': (33.7701, -118.1937),
    }
    
    location_lower = location_name.lower()
    for key, coords in locations.items():
        if key in location_lower:
            return coords
    
    # Default to LA if not found
    return (34.0522, -118.2437)

def calculate_route_plan(start_location, end_location, battery_percent, gas_pedal_level=50, brake_pedal_level=20, weather_conditions='clear', waypoints=None):
    """Calculate route plan with charging recommendations"""
    # Get coordinates
    start_coords = geocode_location(start_location)
    end_coords = geocode_location(end_location)
    
    # Handle waypoints if provided
    all_locations = [start_location]
    all_coords = [start_coords]
    
    if waypoints and len(waypoints) > 0:
        for waypoint in waypoints:
            all_locations.append(waypoint)
            all_coords.append(geocode_location(waypoint))
    
    all_locations.append(end_location)
    all_coords.append(end_coords)
    
    # Calculate total distance across all segments
    total_distance = 0
    for i in range(len(all_coords) - 1):
        segment_distance = haversine_distance(
            all_coords[i][0], all_coords[i][1],
            all_coords[i+1][0], all_coords[i+1][1]
        )
        total_distance += segment_distance
    
    # BMW i3 specs
    max_range_km = BMW_I3_SPECS['range_charging']['range_km']
    usable_capacity_kwh = BMW_I3_SPECS['battery']['usable_capacity_kwh']
    base_consumption_kwh_100km = BMW_I3_SPECS['range_charging']['energy_consumption_kwh_100km']
    
    # Adjust consumption based on driving style (gas pedal level)
    # Higher gas pedal = more aggressive acceleration = higher consumption
    # 0% = very gentle (eco mode), 100% = aggressive (sport mode)
    # Range: -20% to +40% consumption adjustment
    gas_pedal_multiplier = 1.0 + ((gas_pedal_level - 50) / 50) * 0.4  # -0.2 to +0.4
    
    # Adjust consumption based on braking frequency
    # More frequent braking = less regenerative braking efficiency = higher consumption
    # Range: 0% to +15% consumption adjustment
    brake_pedal_multiplier = 1.0 + (brake_pedal_level / 100) * 0.15
    
    # Adjust consumption based on weather conditions
    weather_multipliers = {
        'clear': 1.0,
        'cloudy': 1.05,  # Slight increase due to reduced efficiency
        'rain': 1.15,  # Increased rolling resistance
        'heavy_rain': 1.25,  # Significant increase
        'snow': 1.35,  # Much higher consumption
        'windy': 1.10,  # Headwind resistance
        'extreme_heat': 1.20,  # AC usage
        'cold': 1.25  # Heating usage and battery efficiency loss
    }
    weather_multiplier = weather_multipliers.get(weather_conditions, 1.0)
    
    # Calculate adjusted consumption
    consumption_kwh_100km = base_consumption_kwh_100km * gas_pedal_multiplier * brake_pedal_multiplier * weather_multiplier
    
    # Calculate current range
    current_battery_kwh = (battery_percent / 100) * usable_capacity_kwh
    current_range_km = (current_battery_kwh / consumption_kwh_100km) * 100
    
    # Determine if charging is needed
    needs_charging = current_range_km < total_distance
    
    result = {
        'start_location': start_location,
        'end_location': end_location,
        'waypoints': waypoints if waypoints else [],
        'all_locations': all_locations,
        'start_coords': start_coords,
        'end_coords': end_coords,
        'total_distance_km': round(total_distance, 2),
        'battery_percent': battery_percent,
        'current_range_km': round(current_range_km, 2),
        'max_range_km': max_range_km,
        'needs_charging': needs_charging,
        'charging_stations': [],
        'adjusted_consumption_kwh_100km': round(consumption_kwh_100km, 2),
        'base_consumption_kwh_100km': base_consumption_kwh_100km,
        'driving_factors': {
            'gas_pedal_level': gas_pedal_level,
            'brake_pedal_level': brake_pedal_level,
            'weather_conditions': weather_conditions,
            'gas_pedal_multiplier': round(gas_pedal_multiplier, 3),
            'brake_pedal_multiplier': round(brake_pedal_multiplier, 3),
            'weather_multiplier': round(weather_multiplier, 3)
        }
    }
    
    
    # Find stations along the route (even if charging not needed, show what's available)
    # Search at multiple points along each segment of the route
    search_points = []
    for segment_idx in range(len(all_coords) - 1):
        segment_start = all_coords[segment_idx]
        segment_end = all_coords[segment_idx + 1]
        num_points = 3  # Check 3 points per segment
        for i in range(num_points):
            fraction = (i + 1) / (num_points + 1)
            lat = segment_start[0] + (segment_end[0] - segment_start[0]) * fraction
            lon = segment_start[1] + (segment_end[1] - segment_start[1]) * fraction
            search_points.append((lat, lon))
    
    # Collect all stations near the route
    all_route_stations = []
    seen_stations = set()
    
    for point_lat, point_lon in search_points:
        nearby_stations = station_retriever.find_nearest(point_lat, point_lon, limit=10)
        
        for station in nearby_stations:
            station_id = f"{station.get('latitude')},{station.get('longitude')}"
            if station_id in seen_stations:
                continue
            seen_stations.add(station_id)
            
            station_lat = station.get('latitude')
            station_lon = station.get('longitude')
            
            # Calculate distance from start along the route path
            # Find the closest point on the route to this station
            min_dist_to_route = float('inf')
            dist_along_route = 0
            
            for seg_idx in range(len(all_coords) - 1):
                seg_start = all_coords[seg_idx]
                seg_end = all_coords[seg_idx + 1]
                
                # Calculate distance from station to this segment
                dist_to_seg_start = haversine_distance(station_lat, station_lon, seg_start[0], seg_start[1])
                dist_to_seg_end = haversine_distance(station_lat, station_lon, seg_end[0], seg_end[1])
                seg_length = haversine_distance(seg_start[0], seg_start[1], seg_end[0], seg_end[1])
                
                # Approximate distance to route segment
                min_dist_to_segment = min(dist_to_seg_start, dist_to_seg_end)
                
                # If station is close to this segment, calculate distance along route
                if min_dist_to_segment < 20:  # Within 20km of segment
                    # Calculate distance from start to beginning of this segment
                    dist_to_seg_start_along_route = sum([
                        haversine_distance(all_coords[i][0], all_coords[i][1], 
                                         all_coords[i+1][0], all_coords[i+1][1])
                        for i in range(seg_idx)
                    ])
                    
                    # Approximate position along segment (simplified)
                    dist_along_route = dist_to_seg_start_along_route + (dist_to_seg_start / (dist_to_seg_start + dist_to_seg_end + 0.001)) * seg_length
                    min_dist_to_route = min_dist_to_segment
                    break
            
            # Calculate distance from end
            dist_from_end = total_distance - dist_along_route
            
            # Check if station is reasonably on route (within 15km deviation)
            if min_dist_to_route < 15:
                all_route_stations.append({
                    'name': station.get('station_name', 'Unknown'),
                    'address': station.get('street_address', ''),
                    'latitude': station_lat,
                    'longitude': station_lon,
                    'distance_from_start_km': round(dist_along_route, 2),
                    'distance_from_end_km': round(dist_from_end, 2),
                    'pricing': station.get('ev_pricing', 'N/A'),
                    'rating': station.get('rating', 'N/A'),
                    'charger_type': station.get('charger_type', 'N/A'),
                    'network': station.get('ev_network', 'N/A'),
                    'map_link': f"https://www.google.com/maps/search/?api=1&query={station_lat},{station_lon}"
                })
    
    # Sort by distance from start
    all_route_stations.sort(key=lambda x: x['distance_from_start_km'])
    
    
    if needs_charging:
        # Recommend stations based on when you'll need to charge
        # Charge when you have ~20% buffer (about 105km range remaining)
        safe_range = current_range_km * 0.8  # Use 80% of current range to be safe
        
        # Find stations around the point where you'll need to charge
        recommended_stations = [s for s in all_route_stations 
                               if s['distance_from_start_km'] <= safe_range 
                               and s['distance_from_start_km'] >= safe_range * 0.5]
        
        # If no stations in ideal range, show closest ones
        if not recommended_stations:
            recommended_stations = all_route_stations[:3]
        
        result['charging_stations'] = recommended_stations[:3]  # Top 3 recommendations
        
        # Generate detailed charging plan for the best station
        if recommended_stations:
            best_station = recommended_stations[0]
            
            # Calculate charging needs
            distance_to_station = best_station['distance_from_start_km']
            distance_after_station = total_distance - distance_to_station
            
            # Battery at station arrival
            energy_to_station = (distance_to_station / 100) * consumption_kwh_100km
            battery_at_station = current_battery_kwh - energy_to_station
            battery_percent_at_station = (battery_at_station / usable_capacity_kwh) * 100
            
            # Energy needed to reach destination with 20% buffer
            energy_needed_after = (distance_after_station / 100) * consumption_kwh_100km
            buffer_energy = usable_capacity_kwh * 0.2  # 20% buffer
            target_energy = energy_needed_after + buffer_energy
            
            # How much to charge
            energy_to_add = max(0, target_energy - battery_at_station)
            target_battery_percent = min(100, ((battery_at_station + energy_to_add) / usable_capacity_kwh) * 100)
            
            # Charging time estimate (DC fast charging)
            dc_charging_power = BMW_I3_SPECS['range_charging']['dc_charging_power_kw']
            charging_time_hours = energy_to_add / dc_charging_power
            charging_time_minutes = int(charging_time_hours * 60)
            
            # Cost calculation
            pricing_str = best_station.get('pricing', 'N/A')
            cost_estimate = "N/A"
            if pricing_str != 'N/A' and '$' in pricing_str:
                try:
                    # Extract price per kWh (format: $0.XX/kWh)
                    price_per_kwh = float(pricing_str.replace('$', '').replace('/kWh', ''))
                    total_cost = energy_to_add * price_per_kwh
                    cost_estimate = f"${total_cost:.2f}"
                except:
                    cost_estimate = "N/A"
            
            result['charging_plan'] = {
                'recommended_station': best_station['name'],
                'station_address': best_station['address'],
                'distance_to_station_km': round(distance_to_station, 2),
                'battery_on_arrival_percent': round(battery_percent_at_station, 1),
                'charge_amount_kwh': round(energy_to_add, 2),
                'target_battery_percent': round(target_battery_percent, 1),
                'estimated_charging_time_min': charging_time_minutes,
                'estimated_cost': cost_estimate,
                'distance_after_charging_km': round(distance_after_station, 2),
                'steps': [
                    f"1. Drive {round(distance_to_station, 1)} km to {best_station['name']}",
                    f"2. Arrive with approximately {round(battery_percent_at_station, 1)}% battery",
                    f"3. Charge {round(energy_to_add, 1)} kWh (to {round(target_battery_percent, 1)}%)",
                    f"4. Estimated charging time: {charging_time_minutes} minutes",
                    f"5. Estimated cost: {cost_estimate}",
                    f"6. Continue {round(distance_after_station, 1)} km to destination",
                    f"7. Arrive at destination with ~20% battery buffer"
                ]
            }
    else:
        # Even if charging not needed, show stations along route for reference
        result['charging_stations'] = all_route_stations[:5]  # Show up to 5 stations
        result['charging_plan'] = None
    
    
    
    return result

@app.route('/api/plan-route', methods=['POST'])
def plan_route():
    """API endpoint for route planning"""
    try:
        data = request.json
        start = data.get('start_location')
        end = data.get('end_location')
        battery = float(data.get('battery_percent', 100))
        gas_pedal = float(data.get('gas_pedal_level', 50))
        brake_pedal = float(data.get('brake_pedal_level', 20))
        weather = data.get('weather_conditions', 'clear')
        
        if not start or not end:
            return jsonify({'error': 'Start and end locations required'}), 400
        
        if battery < 0 or battery > 100:
            return jsonify({'error': 'Battery percent must be between 0 and 100'}), 400
        
        if gas_pedal < 0 or gas_pedal > 100:
            return jsonify({'error': 'Gas pedal level must be between 0 and 100'}), 400
        
        if brake_pedal < 0 or brake_pedal > 100:
            return jsonify({'error': 'Brake pedal level must be between 0 and 100'}), 400
        
        waypoints = data.get('waypoints', [])
        if not isinstance(waypoints, list):
            waypoints = []
        
        result = calculate_route_plan(start, end, battery, gas_pedal, brake_pedal, weather, waypoints)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vehicle-specs', methods=['GET'])
def get_vehicle_specs():
    """Get BMW i3 specifications"""
    return jsonify(BMW_I3_SPECS)

@app.route('/api/ask-llm', methods=['POST'])
def ask_llm():
    """Process natural language query about trip and charging needs"""
    try:
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Use Mistral to parse the query and extract trip details
        system_prompt = f"""You are an EV trip planning assistant. Extract comprehensive trip information from user queries.

BMW i3 Specs:
- Max Range: {BMW_I3_SPECS['range_charging']['range_km']} km
- Battery Capacity: {BMW_I3_SPECS['battery']['usable_capacity_kwh']} kWh
- Consumption: {BMW_I3_SPECS['range_charging']['energy_consumption_kwh_100km']} kWh/100km

Extract from the user's query:
1. Start location (city/area name) - REQUIRED
2. End location (city/area name) - REQUIRED
3. Waypoints/intermediate stops (array of location names) - OPTIONAL, extract if user mentions "via", "through", "stopping at", "visiting", etc.
4. Current battery percentage (if mentioned, otherwise assume 80%)
5. Weather conditions - extract from words like: "rain", "rainy", "snow", "snowy", "sunny", "clear", "cloudy", "windy", "hot", "cold", "extreme heat", "heavy rain", etc.
   Map to one of: "clear", "cloudy", "rain", "heavy_rain", "snow", "windy", "extreme_heat", "cold"
6. Driving style - extract from words like: "aggressive", "fast", "sport", "gentle", "eco", "slow", "normal", "calm"
   - aggressive/fast/sport = high gas pedal (70-90%)
   - gentle/eco/slow/calm = low gas pedal (20-40%)
   - normal = medium gas pedal (50%)
7. Braking frequency - extract from words like: "frequent braking", "stop and go", "city driving", "highway driving"
   - city/stop and go/frequent = high brake pedal (40-60%)
   - highway/smooth = low brake pedal (10-20%)
   - normal = medium brake pedal (20%)

Respond in JSON format:
{{
  "start": "location name",
  "end": "location name",
  "waypoints": ["location1", "location2"] or [] if none,
  "battery": number (0-100),
  "weather": "clear" | "cloudy" | "rain" | "heavy_rain" | "snow" | "windy" | "extreme_heat" | "cold",
  "gas_pedal": number (0-100, default 50),
  "brake_pedal": number (0-100, default 20)
}}

Examples:
- "I'm going from LA to San Francisco via Santa Barbara with 50% battery in the rain"
  -> {{"start": "Los Angeles", "end": "San Francisco", "waypoints": ["Santa Barbara"], "battery": 50, "weather": "rain", "gas_pedal": 50, "brake_pedal": 20}}
  
- "Driving aggressively from Hollywood to Long Beach, it's snowing, 30% battery"
  -> {{"start": "Hollywood", "end": "Long Beach", "waypoints": [], "battery": 30, "weather": "snow", "gas_pedal": 80, "brake_pedal": 20}}

- "Eco mode from Pasadena to Santa Monica via Beverly Hills, sunny day, 80% charge"
  -> {{"start": "Pasadena", "end": "Santa Monica", "waypoints": ["Beverly Hills"], "battery": 80, "weather": "clear", "gas_pedal": 30, "brake_pedal": 20}}

If you cannot extract start and end locations, respond with {{"error": "explanation"}}"""

        # Call Mistral API
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistral-large-latest",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(MISTRAL_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return jsonify({'error': f'LLM API error: {response.text}'}), 500
        
        llm_response = response.json()
        extracted_data = json.loads(llm_response['choices'][0]['message']['content'])
        
        if 'error' in extracted_data:
            return jsonify({
                'success': False,
                'message': f"I couldn't understand your query. {extracted_data['error']}",
                'suggestion': "Try something like: 'I'm going from Hollywood to Long Beach with 30% battery'"
            })
        
        # Extract additional parameters if provided, otherwise use defaults
        gas_pedal = float(extracted_data.get('gas_pedal', 50))
        brake_pedal = float(extracted_data.get('brake_pedal', 20))
        weather = extracted_data.get('weather', 'clear')
        waypoints = extracted_data.get('waypoints', [])
        
        # Validate waypoints
        if not isinstance(waypoints, list):
            waypoints = []
        
        # Calculate route plan with extracted data
        route_plan = calculate_route_plan(
            extracted_data['start'],
            extracted_data['end'],
            extracted_data['battery'],
            gas_pedal,
            brake_pedal,
            weather,
            waypoints
        )
        
        # Build route description
        route_description = f"{extracted_data['start']}"
        if waypoints:
            route_description += " → " + " → ".join(waypoints)
        route_description += f" → {extracted_data['end']}"
        
        # Build factors description
        factors_desc = []
        if weather != 'clear':
            weather_labels = {
                'clear': '☀️ Clear',
                'cloudy': '☁️ Cloudy',
                'rain': '🌧️ Rainy',
                'heavy_rain': '⛈️ Heavy Rain',
                'snow': '❄️ Snowy',
                'windy': '💨 Windy',
                'extreme_heat': '🔥 Extreme Heat',
                'cold': '🧊 Cold'
            }
            factors_desc.append(f"Weather: {weather_labels.get(weather, weather)}")
        
        if gas_pedal > 60:
            factors_desc.append("Driving style: Aggressive")
        elif gas_pedal < 40:
            factors_desc.append("Driving style: Eco/Gentle")
        
        if brake_pedal > 40:
            factors_desc.append("City/Stop-and-go driving")
        
        factors_text = "\n".join(factors_desc) if factors_desc else ""
        
        # Generate natural language response
        if route_plan['needs_charging']:
            if route_plan.get('charging_plan'):
                plan = route_plan['charging_plan']
                message = f"""🔋 **Yes, you'll need to charge!**

📍 **Your Trip:** {route_description}
📏 **Total Distance:** {route_plan['total_distance_km']} km
🔋 **Current Range:** {route_plan['current_range_km']} km (at {extracted_data['battery']}% battery)
{f'📊 **Conditions:** {factors_text}' if factors_text else ''}

⚡ **Recommended Charging Stop:**
**{plan['recommended_station']}**
📍 {plan['station_address']}

**Charging Details:**
• Drive {plan['distance_to_station_km']} km to the station
• Arrive with ~{plan['battery_on_arrival_percent']}% battery
• Charge {plan['charge_amount_kwh']} kWh (to {plan['target_battery_percent']}%)
• Estimated time: {plan['estimated_charging_time_min']} minutes
• Estimated cost: {plan['estimated_cost']}

Then continue {plan['distance_after_charging_km']} km to your destination!"""
            else:
                message = f"Yes, you'll need to charge. Distance: {route_plan['total_distance_km']} km, Your range: {route_plan['current_range_km']} km"
        else:
            message = f"""✅ **No charging needed!**

📍 **Your Trip:** {route_description}
📏 **Total Distance:** {route_plan['total_distance_km']} km
🔋 **Your Range:** {route_plan['current_range_km']} km (at {extracted_data['battery']}% battery)
{f'📊 **Conditions:** {factors_text}' if factors_text else ''}

You have plenty of range to make it! You'll arrive with approximately {round((route_plan['current_range_km'] - route_plan['total_distance_km']) / route_plan['max_range_km'] * 100)}% battery remaining."""
        
        return jsonify({
            'success': True,
            'message': message,
            'route_plan': route_plan,
            'extracted_data': extracted_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
