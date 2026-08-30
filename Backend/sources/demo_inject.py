import sys
import json
import urllib.request
import urllib.error

# Your live Render backend URL
API_URL = "https://sih-backend-jgez.onrender.com/screen"

def print_banner():
    print("==========================================================")
    print("🚀 ISRO GROUND STATION - ML TELEMETRY INJECTION TOOL 🚀")
    print("==========================================================")

def main():
    if len(sys.argv) < 3:
        print("Usage: python demo_inject.py <value_0h> <value_24h>")
        print("Example: python demo_inject.py 12.0 45.0")
        return

    try:
        val_0 = float(sys.argv[1])
        val_24 = float(sys.argv[2])
    except ValueError:
        print("Error: Please provide valid numbers for telemetry values.")
        return

    print_banner()
    print(f"📡 Injecting live telemetry to cloud backend...")
    print(f"   ► Component : DEMO_TEST_PART")
    print(f"   ► Value @ 0h : {val_0} µA")
    print(f"   ► Value @ 24h: {val_24} µA\n")

    # Construct the JSON payload for the ML Engine
    payload = {
        "readings": [
            {
                "component_id": "DEMO_TEST_PART",
                "lot_id": "LVM3_STAGE_02",
                "parameter": "Iddq",
                "value_0h": val_0,
                "value_24h": val_24,
                "datasheet_max": 55.0, # Standard LVM3 limit
                "datasheet_min": 0.0
            }
        ],
        "z_score_threshold": 3.5,
        "safety_margin": 0.85
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            
            # Parse the response
            verdict = result["verdicts"][0]
            decision = verdict["final_decision"]
            summary = verdict["qa_summary"]
            
            print("==========================================================")
            print("🧠 ML ENGINE VERDICT")
            print("==========================================================")
            
            if decision == "PASS":
                print(f"✅ STATUS: {decision}")
            elif decision == "WATCH":
                print(f"⚠️ STATUS: {decision}")
            else:
                print(f"❌ STATUS: {decision}")
                
            print(f"📝 ANALYSIS: {summary}")
            print("==========================================================\n")
            
    except urllib.error.URLError as e:
        print(f"🚨 Network Error: Failed to reach backend at {API_URL}")
        print(f"Details: {e}")

if __name__ == "__main__":
    main()
