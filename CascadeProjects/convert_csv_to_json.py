import csv
import json
import random

def convert_csv_to_json(csv_file_path, json_file_path):
    stations = []
    
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Parse address to attempt to extract city/state/zip if possible, 
            # but for now we'll just use the full address string.
            # The CSV address format is "Street, City, Country" or similar.
            
            station = {
                "id": row["Station ID"],
                "station_name": f"{row['Station Operator']} Station - {row['Station ID']}",
                "street_address": row["Address"],
                "city": row["Address"].split(",")[1].strip() if "," in row["Address"] else "Unknown",
                "state": "Unknown", # CSV doesn't seem to have state codes consistently
                "zip": "Unknown",
                "latitude": float(row["Latitude"]),
                "longitude": float(row["Longitude"]),
                "access_days_time": row["Availability"],
                "ev_network": row["Station Operator"],
                "ev_connector_types": [c.strip() for c in row["Connector Types"].split(",")],
                "ev_pricing": f"${row['Cost (USD/kWh)']}/kWh",
                "charger_type": row["Charger Type"],
                "charging_capacity_kw": row["Charging Capacity (kW)"],
                "usage_stats": row["Usage Stats (avg users/day)"],
                "renewable_energy": row["Renewable Energy Source"],
                "rating": row["Reviews (Rating)"]
            }
            stations.append(station)
            
    output_data = {"fuel_stations": stations}
    
    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(output_data, jsonfile, indent=4)
    
    print(f"Successfully converted {len(stations)} stations to {json_file_path}")

if __name__ == "__main__":
    convert_csv_to_json("detailed_ev_charging_stations.csv", "converted_detailed_stations.json")
