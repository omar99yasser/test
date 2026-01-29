# BMW i3 Route Planner - Quick Start Guide

## ✅ Backend is Running Successfully!

The API test shows the backend is working:
- Server: http://localhost:5000
- Status: ✅ Running
- Stations Loaded: 5,000

## 🔧 How to Use the Web Interface

### Option 1: Use Python's Built-in Server (Recommended)
```bash
# In a new terminal, run:
cd /Users/omaryasser/CascadeProjects
python3 -m http.server 8000
```

Then open in your browser:
**http://localhost:8000/ev_route_planner.html**

### Option 2: Test with curl (Command Line)
```bash
curl -X POST http://localhost:5000/api/plan-route \
  -H "Content-Type: application/json" \
  -d '{"start_location": "Hollywood", "end_location": "Long Beach", "battery_percent": 30}'
```

## 📝 Example Test Results

**Route:** Hollywood → Santa Monica (80% battery)
- Distance: 17.05 km
- Current Range: 369.79 km  
- Status: ✅ No charging needed

**Try a longer route with low battery:**
- Start: Hollywood
- End: Long Beach
- Battery: 30%
- Expected: Will recommend charging stations!

## 🛑 To Stop the Server
Press `Ctrl+C` in the terminal running the backend.
