import requests
import json

def test_hindi_chat():
    url = "http://127.0.0.1:8000/chat"
    payload = {
        "user_input": "Mujhe bahut tej bukhar aur sir dard ho raha hai" # "I have high fever and headache" in Hindi
    }
    
    try:
        response = requests.post(url, json=payload)
        print("Status Code:", response.status_code)
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error connecting to server: {e}. Make sure api_server.py is running.")

if __name__ == "__main__":
    test_hindi_chat()
