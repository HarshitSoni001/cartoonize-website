import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
from huggingface_hub import snapshot_download
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the model (downloads once)
model_path = snapshot_download("sayakpaul/whitebox-cartoonizer")
loaded_model = tf.saved_model.load(model_path)
concrete_func = loaded_model.signatures["serving_default"]

# Helper functions
def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image

def preprocess_image(image):
    image = resize_crop(image)
    image = image.astype(np.float32) / 127.5 - 1
    image = np.expand_dims(image, axis=0)
    image = tf.constant(image)
    return image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            # Process the image
            try:
                pil_image = Image.open(input_path).convert("RGB")
                image_np = np.array(pil_image)
                image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                preprocessed_image = preprocess_image(image)
                result = concrete_func(preprocessed_image)["final_output:0"]
                output = (result[0].numpy() + 1.0) * 127.5
                output = np.clip(output, 0, 255).astype(np.uint8)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                output_image = Image.fromarray(output)
                
                output_filename = 'cartoonized_' + filename
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                output_image.save(output_path)
                
                return render_template('index.html', input_image=filename, output_image=output_filename)
            except Exception as e:
                return render_template('index.html', error=str(e))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
