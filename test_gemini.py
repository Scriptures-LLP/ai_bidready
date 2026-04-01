import asyncio
import httpx

async def test():
    api_key = "AIzaSyCDQ0PeW3GKx6NK2PiHViRFBnvQJGPU1ck"
    model_name = "gemini-flash-lite-latest"
    # Try gemini-1.5-flash as fallback if it fails
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": "Analyze this architectural blueprint. Return a JSON object with a single key 'building_bbox' containing [ymin, xmin, ymax, xmax] coordinates (normalized 0.0 to 1.0). Return ONLY the JSON object."}
            ]
        }],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    print(f"Testing model: {model_name}")
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.post(url, json=payload)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                print(f"Response: {resp.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No text')}")
            else:
                print(f"Error Response: {resp.text}")
        except Exception as e:
            print(f"Caught Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test())
