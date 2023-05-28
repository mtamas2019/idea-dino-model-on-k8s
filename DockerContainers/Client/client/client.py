import os
import requests
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, render_template, request
import cv2
import base64

app = Flask(__name__)

# Variables
api_url = str(os.getenv('API_URL', 'http://localhost:8080/inference'))
port = int(os.getenv('PORT', 80))
threshold=float(os.getenv('TRESHOLD', 0.5))


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded.')

        file = request.files['file']

        # Check if the file is an image
        if file.filename == '':
            return render_template('index.html', error='No file selected.')

        if not allowed_file(file.filename):
            return render_template('index.html', error='Unsupported file type.')

        # Save the uploaded image temporarily
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)
        image_path = os.path.join(upload_folder, file.filename)
        file.save(image_path)

        # Perform inference on the image
        response = perform_inference(image_path)

        # Process the response and overlay on the image
        annotated_image = overlay_boxes(image_path, response,threshold)

        # Display the annotated image in the UI
        return render_template('result.html', image_path=image_path, annotated_image=annotated_image)

    return render_template('index.html')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def perform_inference(image_path):
    # Create the payload with the image file
    files = {'image': open(image_path, 'rb')}

    # Send the POST request to the API
    response = requests.post(api_url, files=files)

    if response.status_code == 200:
        # Process the JSON response
        result = response.json()
        return result

    return {'error': 'Inference failed.'}


import base64

def overlay_boxes(image_path, response, threshold):
    # Load the image
    image = Image.open(image_path)
    image_np = np.array(image)

    # Extract the bounding boxes, labels, and scores
    boxes = response['boxes']
    labels = response.get('labels', [])  # Assuming labels are available, otherwise empty list
    scores = response.get('scores', [])  # Assuming scores are available, otherwise empty list

    # Overlay bounding boxes and labels on the image for predictions above the threshold
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            xmin, ymin, xmax, ymax = box

            # Adjust the bounding box coordinates to fit the image dimensions
            xmin = int(xmin * image.width)
            ymin = int(ymin * image.height)
            xmax = int(xmax * image.width)
            ymax = int(ymax * image.height)

            # Draw the bounding box rectangle
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(image_np, pt1, pt2, (255, 0, 0), 2)

            select_mask = score > threshold
            with open('util/coco_id2name.json') as f:
              id2name = json.load(f)
              id2name = {int(k): v for k, v in id2name.items()}

            box_label = id2name[int(label)]  # Directly use the label value


            # Add the label text and score
            label_text = f'{box_label} ({score:.2f})'
            cv2.putText(image_np, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convert the NumPy array back to PIL Image
    annotated_image = Image.fromarray(image_np)

    # Convert PIL Image to base64 string for HTML rendering
    buffered = BytesIO()
    annotated_image.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)
