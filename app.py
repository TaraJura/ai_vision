import cv2
import base64
import openai
import time
import os

def get_first_active_camera(max_devices=3):
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

def capture_webcam_image():
    cam_index = get_first_active_camera()
    if cam_index is None:
        raise IOError("No active camera found.")
    
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera {cam_index}")
    
    time.sleep(2)  # let camera warm up
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise IOError("Failed to capture image")
        
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def analyze_image(base64_image, client):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {e}"

def main():
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    openai.api_key = api_key

    try:
        print("Capturing image...")
        base64_image = capture_webcam_image()
        print("Analyzing image...")
        result = analyze_image(base64_image, openai)
        print("\nAnalysis result:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()