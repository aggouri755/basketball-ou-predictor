import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    """Convert image to grayscale and apply threshold for better OCR accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

import easyocr

reader = easyocr.Reader(['en'])

def extract_text(image):
    """Extract text from an image using EasyOCR with better filtering."""
    processed_img = preprocess_image(image)
    extracted_text = reader.readtext(processed_img, detail=0)

    # Keep only text lines that contain numbers (likely scores)
    filtered_text = [line for line in extracted_text if any(char.isdigit() for char in line)]

    return " ".join(filtered_text)
def predict_over_under(stats):
    """Calculate Over/Under prediction based on extracted stats."""
    try:
        lines = stats.split("\n")
        score_line = [line for line in lines if ":" in line and len(line.split(":")) == 2]
        
        if score_line:
            scores = score_line[0].split(":")
            team1_score = int(scores[0].strip())
            team2_score = int(scores[1].strip())
            total_score = team1_score + team2_score
            
            projected_total = (total_score / 3) * 4  # Adjusting for game pace
            over_under_line = 170  # Dynamic line can be adjusted based on past trends
            
            prediction = "Over" if projected_total > over_under_line else "Under"
            return f"Projected Total: {projected_total:.1f}, Over/Under Prediction: {prediction}"
        else:
            return "Unable to extract score information. Please try another image."
    except Exception as e:
        return f"Error processing stats: {str(e)}"

# Streamlit Web App
st.title("Basketball Over/Under Prediction")
st.write("Upload a screenshot of a live game, and the AI will predict Over/Under.")

uploaded_file = st.file_uploader("Upload Game Screenshot", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Extracting game stats...")
    extracted_text = extract_text(image_np)
    st.text(extracted_text)
    
    st.write("Calculating Over/Under Prediction...")
    prediction_result = predict_over_under(extracted_text)
    st.write(prediction_result)
