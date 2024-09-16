import os
from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
from collections import Counter
import cv2

# Initialize the Flask app
app = Flask(__name__)

# Load the YOLO model
model = YOLO('models/yolov8n_tralicci.pt')

# Ensure the uploads and results directories exist
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Get the uploaded image
    file = request.files['image']
    
    # Save the image to the upload folder
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Perform inference on the image
    results = model(image_path)

    # Extract detected classes and count them
    detected_classes = results[0].boxes.cls.cpu().numpy()
    class_counts = Counter(detected_classes)

    # Get class names
    class_names = model.names

    # Load the original image for drawing
    img = cv2.imread(image_path)

    # Draw bounding boxes and labels on the image
    for box in results[0].boxes:
        # Box coordinates and class ID
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
        class_id = int(box.cls)

        # Draw the rectangle (bounding box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label the box with the class name
        label = f"{class_names[class_id]}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the result image
    result_image_path = os.path.join(RESULT_FOLDER, file.filename)
    cv2.imwrite(result_image_path, img)

    # Format the detections as a dictionary
    detections = []
    for class_id, count in class_counts.items():
        detections.append({
            "class": class_names[int(class_id)],
            "count": count
        })

    # Return the detections along with the result image
    response = {
        "detections": detections,
        "total_objects": sum(class_counts.values()),
        "result_image": result_image_path
    }
    return jsonify(response)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
