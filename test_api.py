"""
Test script for NutriSnap API - DETAILED OUTPUT
"""

import requests
import base64
import os
import json

BASE_URL = 'http://localhost:5000'

print("="*60)
print("🧪 NUTRISNAP API TEST - DETAILED")
print("="*60)

# Test 1: Health Check
print("\n1️⃣ Testing /health...")
try:
    response = requests.get(f'{BASE_URL}/health')
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Model Loaded: {data['model_loaded']}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Get Classes
print("\n2️⃣ Testing /classes...")
try:
    response = requests.get(f'{BASE_URL}/classes')
    data = response.json()
    print(f"   Total Classes: {data['total']}")
    print(f"   Classes: {', '.join(data['classes'])}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Detection with FULL OUTPUT
print("\n3️⃣ Testing /detect...")
test_folder = 'test_images'
if os.path.exists(test_folder):
    images = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if images:
        image_path = os.path.join(test_folder, images[0])
        print(f"   📷 Image: {image_path}")
        print("-"*60)
        
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        response = requests.post(
            f'{BASE_URL}/detect',
            json={'image': image_base64}
        )
        data = response.json()
        
        if data['success']:
            print(f"\n   ✅ SUCCESS!")
            print(f"\n   📋 SUMMARY:")
            print(f"      Unique Ingredients: {data['detected_ingredients']}")
            print(f"      Count: {data['count']}")
            print(f"      Total Detections: {data['total_detections']}")
            
            print(f"\n   🔍 DETAILED DETECTIONS:")
            for i, det in enumerate(data['detections'], 1):
                confidence_percent = det['confidence'] * 100
                print(f"      {i}. {det['name']}: {confidence_percent:.1f}% confidence")
                if 'bounding_box' in det:
                    box = det['bounding_box']
                    print(f"         Box: x1={box['x1']}, y1={box['y1']}, x2={box['x2']}, y2={box['y2']}")
            
            print(f"\n   📦 FULL JSON RESPONSE:")
            print(json.dumps(data, indent=4))
        else:
            print(f"   ❌ Error: {data.get('error')}")
    else:
        print("   ⚠️ No images in test_images folder")
else:
    print("   ⚠️ test_images folder not found")

print("\n" + "="*60)
print("✅ Test completed!")
print("="*60)