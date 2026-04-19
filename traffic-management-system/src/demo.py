import cv2
import numpy as np
import time

# --- CONFIGURATION ---
# Ensure your video file is named exactly "traffic.mp4" in the folder
VIDEO_PATH = "traffic.mp4" 

# --- LOAD YOLO (The AI Model) ---
print("Loading AI Model... please wait.")
try:
    # Load Weights and Config
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    # Load Names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"Error loading files: {e}")
    print("Please check that yolov3.weights, yolov3.cfg, and coco.names are in the folder.")
    exit()

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# --- START VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)
font = cv2.FONT_HERSHEY_PLAIN

# For logging timing (to prevent flooding the terminal)
last_log_time = time.time()

if not cap.isOpened():
    print(f"Error: Could not open video file '{VIDEO_PATH}'. Check the filename!")
    exit()

print("System Started. Press 'ESC' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("--- Video Loop Restarting ---")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    height, width, channels = frame.shape

    # --- DETECT OBJECTS ---
    # FIX: "crop" must be lowercase
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Remove double boxes (Non-Max Suppression)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # --- LOGIC & COUNTING ---
    vehicle_count = 0
    emergency_detected = False
    
    # Only count these vehicle types
    vehicle_types = ["car", "motorbike", "bus", "truck"]

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            
            if label in vehicle_types:
                vehicle_count += 1
                color = (0, 0, 255) # Red for normal traffic
                
                # --- SIMULATE EMERGENCY LOGIC ---
                # We treat 'truck' or 'bus' as an Ambulance for the demo
                if label == "truck" or label == "bus":
                    emergency_detected = True
                    label = "EMERGENCY" 
                    color = (0, 255, 0) # Green

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), font, 1, color, 2)

    # --- DECISION MATRIX (The "Brain") ---
    timer = 0
    status = "RED"
    status_color = (0, 0, 255) # Red

    if emergency_detected:
        timer = 999
        status = "EMERGENCY OVERRIDE - GREEN"
        status_color = (0, 255, 0)
    elif vehicle_count < 15: 
        timer = 30
        status = "Low Density - GREEN"
        status_color = (0, 255, 0)
    elif vehicle_count >= 15 and vehicle_count < 30:
        timer = 45
        status = "Med Density - GREEN"
        status_color = (0, 255, 0)
    else:
        timer = 60
        status = "High Density - GREEN"
        status_color = (0, 255, 0)

    # --- TERMINAL LOGGING (Matches your screenshot) ---
    # Log every 1.5 seconds to keep it readable
    if time.time() - last_log_time > 1.5:
        print(f"Vehicle Count: {vehicle_count} | Decision: {status} for {timer} seconds")
        last_log_time = time.time()

    # --- VISUAL DASHBOARD ---
    # Draw black background box
    cv2.rectangle(frame, (0, 0), (450, 120), (0, 0, 0), -1)
    # Draw Text
    cv2.putText(frame, f"Count: {vehicle_count}", (10, 30), font, 2, (255, 255, 255), 2)
    cv2.putText(frame, f"Signal: {status}", (10, 70), font, 1.3, status_color, 2)
    cv2.putText(frame, f"Timer: {timer}s", (10, 105), font, 1.5, status_color, 2)

    cv2.imshow("Smart Traffic Control System", frame)
    
    key = cv2.waitKey(1)
    if key == 27: # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()