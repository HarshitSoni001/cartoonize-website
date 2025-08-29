# Cartoonize Website

A Flask-based web app that cartoonizes uploaded images using an AI model (White-box Cartoonization).

## Features
- Upload images (JPG, PNG, WEBP).
- Automatically cartoonize and display results.
- Simple, user-friendly interface with CSS styling.

## Screenshots
(Add images of your app here, e.g., upload screenshots via GitHub and link them.)

## Requirements
- Python 3.8+
- Dependencies (install with `pip install -r requirements.txt`):
  - Flask
  - tensorflow
  - huggingface_hub
  - opencv-python-headless
  - numpy
  - pillow
  - werkzeug

## How to Run Locally
1. Clone the repo: `git clone https://github.com/yourusername/cartoonize-website.git`
2. Create and activate a virtual environment: `python -m venv venv` then `venv\Scripts\activate` (Windows).
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `python app.py`
5. Open http://127.0.0.1:5000 in your browser and upload an image.

## Deployment
(Optional: Add instructions for Heroku or other platforms if you deploy it.)

## License
MIT License (or choose another).
