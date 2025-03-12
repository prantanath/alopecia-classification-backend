import os
import base64
import pandas as pd
import numpy as np
import cv2
import os
from flask_cors import CORS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress unnecessary warnings
import tensorflow as tf
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load trained model
model = tf.keras.models.load_model("mobilenet.h5")

# Load class names
class_names = ['Alopecia_Areata','Alopecia_Totalis', 'Androgenetic_Alopecia']

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def calculate_entropy(image):
    """Compute Shannon entropy manually for an image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()  # Normalize histogram
    hist = hist[hist > 0]  # Remove zero values to avoid log(0)
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def detect_text_edges(image):
    """Detect strong text-like edges and aspect ratios to filter out screenshots."""
    edges = cv2.Canny(image, 100, 200)  # Edge detection
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])  # Ratio of edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_contours = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h != 0 else 0
        if 1.5 < aspect_ratio < 5 and 10 < cv2.contourArea(cnt) < 1000:  # Typical text aspect ratio
            text_contours += 1
    return edge_density, text_contours

def detect_hair_and_patches(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return False, "Warning: Unable to load image"
    
    # Convert to grayscale for face detection and entropy calculation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with stricter parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
    
    # Convert to HSV for hair/skin detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Even stricter hair color range
    lower_hair = np.array([0, 30, 30], dtype=np.uint8)  # Avoid dark UI elements
    upper_hair = np.array([180, 140, 140], dtype=np.uint8)  # Natural hair tones only

    # Skin color range for bald patches/heads
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create masks
    hair_mask = cv2.inRange(hsv, lower_hair, upper_hair)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5, 5), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    hair_contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    skin_contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate areas
    total_area = img.shape[0] * img.shape[1]
    hair_area = sum(cv2.contourArea(cnt) for cnt in hair_contours)
    skin_area = sum(cv2.contourArea(cnt) for cnt in skin_contours)

    # Head validation
    if len(faces) == 0:
        # Stricter circular shape detection for bald heads
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                                   param1=100, param2=60, minRadius=50, maxRadius=200)
        if circles is None:
            # Stricter hair/skin requirement without head detection
            if hair_area / total_area < 0.15 and skin_area / total_area < 0.3:
                # Enhanced screenshot detection
                img_entropy = calculate_entropy(gray)
                edge_density, text_contours = detect_text_edges(gray)
                hsv_std = np.std(hsv, axis=(0, 1))

                # Stricter screenshot criteria
                if (img_entropy > 6.5 or edge_density > 0.04 or text_contours > 15) and hsv_std[2] < 30:
                    return False, "Warning: Image appears to be a synthetic screenshot or random object"
                return False, "Warning: No head detected - insufficient hair or skin"
    
    # Detect bald patches if hair is present
    bald_patches = 0
    if len(hair_contours) > 0:
        hair_region = cv2.dilate(hair_mask, kernel, iterations=3)
        patch_mask = cv2.bitwise_and(skin_mask, hair_region)
        patch_contours, _ = cv2.findContours(patch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        patch_threshold = 200
        bald_patches = len([cnt for cnt in patch_contours if cv2.contourArea(cnt) > patch_threshold])

    return True, {
        "hair_detected": len(hair_contours) > 0,
        "skin_detected": len(skin_contours) > 0,
        "bald_patches": bald_patches
    }


# Preprocess uploaded image
def preprocess_head_numpy(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0 
    return np.expand_dims(img, axis=0)

# Prediction & marking function
def predict_and_mark_affected_skin(image_path):
    img_array = preprocess_head_numpy(image_path)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred[0])
    pred_class = class_names[class_idx]
    confidence = round(pred[0][class_idx] * 100, 2)

    # Mark affected area
    head_image = img_array[0]
    hsv = cv2.cvtColor((head_image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marked_image = (head_image * 255).astype(np.uint8).copy()
    cv2.drawContours(marked_image, contours, -1, (255, 0, 0), 2)

    # Convert marked image to base64
    _, buffer = cv2.imencode(".png", marked_image)
    marked_image_base64 = base64.b64encode(buffer).decode("utf-8")

    return pred_class, confidence, marked_image_base64

@app.route("/")
def home():
    return "Flask API is running!"

# API Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    # Check for hair and skin
    result, info = detect_hair_and_patches(file_path)
    if not result:
        return jsonify({"error": info}), 400

    # Predict and mark
    pred_class, confidence, marked_image_base64 = predict_and_mark_affected_skin(file_path)

    return jsonify({
        "predicted_class": pred_class,
        "confidence": round(float(confidence),2),  # Convert float32 to Python float
        "marked_image": marked_image_base64
    })

if __name__ == "__main__":
    app.run()
