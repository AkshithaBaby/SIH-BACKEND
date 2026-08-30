import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("\n--- Health Check ---")
r1 = client.get('/health')
print(r1.json())

with open('synthetic_readings.json') as f:
    data = json.load(f)

print("\n--- Drift Train ---")
r2 = client.post('/drift/train', json=data)
print(r2.json())

print("\n--- Screen Batch ---")
r3 = client.post('/screen', json={'readings': data, 'z_score_threshold': 3.5, 'safety_margin': 0.85})
res3 = r3.json()
print("Metrics:", res3.get('metrics'))
print("Sample Verdict:", res3.get('verdicts')[0] if res3.get('verdicts') else "No verdicts")
