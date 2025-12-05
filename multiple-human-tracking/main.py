import cv2
from ultralytics import YOLO
import os
import argparse
from pathlib import Path

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Multiple Human Tracking using YOLO')
parser.add_argument('video_path', type=str, help='Path to the input video file')
args = parser.parse_args()

model = YOLO("yolo11n.pt")

video_path = args.video_path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# Get video properties for output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output path with same filename
input_filename = Path(video_path).name
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, input_filename)

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Processing video: {video_path}")
print(f"Output will be saved to: {output_path}")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    results = model.track(frame, persist=True, classes=[0])  
    # Class 0 = Person
    result = results[0]

    annotated = result.plot()
    
    # Write frame to output video
    out.write(annotated)
    
    cv2.imshow("Multiple Human Tracking (YOLO)", annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Video saved successfully to: {output_path}")
