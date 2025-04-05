import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from config import *
from main import process_image

# Set page configuration
st.set_page_config(page_title="KITTI Lane Detection", layout="wide")

# Title and description
st.title("🚗 Lane Line Detection with KITTI Dataset")

st.markdown("""
## About this Project
This application demonstrates lane line detection using the KITTI Road Dataset.
The system uses computer vision techniques to identify and highlight lane markings in road images.

The detection pipeline includes:
1. Color filtering (white and yellow lanes)
2. Edge detection using Canny algorithm
3. Region of interest selection
4. Hough line detection
5. Line averaging and drawing
""")

# Sidebar for dataset navigation and parameters
st.sidebar.header("KITTI Dataset Navigation")

# Dataset selection
dataset_type = st.sidebar.radio(
    "Select dataset type:",
    ("Training", "Testing")
)

# Define dataset paths
if dataset_type == "Training":
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "training", "image_2")
else:
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "testing", "image_2")

# Get list of images in the selected dataset
image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg'))]

# Option to select image or upload custom image
image_option = st.sidebar.radio(
    "Choose image source:",
    ("Select from Dataset", "Upload Custom Image")
)

# Parameter tuning in sidebar
st.sidebar.header("Parameter Tuning")

# Canny Edge Detection Parameters
st.sidebar.subheader("Edge Detection")
canny_low = st.sidebar.slider("Canny Low Threshold", 10, 100, CANNY_LOW_THRESHOLD)
canny_high = st.sidebar.slider("Canny High Threshold", 50, 200, CANNY_HIGH_THRESHOLD)

# Hough Transform Parameters
st.sidebar.subheader("Hough Transform")
hough_threshold = st.sidebar.slider("Hough Threshold", 10, 50, HOUGH_THRESHOLD)
hough_min_line_length = st.sidebar.slider("Min Line Length", 50, 150, HOUGH_MIN_LINE_LENGTH)
hough_max_line_gap = st.sidebar.slider("Max Line Gap", 50, 250, HOUGH_MAX_LINE_GAP)

# Line Parameters
st.sidebar.subheader("Line Detection")
min_slope = st.sidebar.slider("Minimum Slope", 0.1, 1.0, MIN_SLOPE)
smoothing_factor = st.sidebar.slider("Smoothing Factor", 0.1, 1.0, LINE_SMOOTHING_FACTOR)

# Main content area
st.header("Lane Detection Results")

# Function to process image with user parameters
def process_image_with_params(image):
    # Declare globals first
    global CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD
    global HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP
    global MIN_SLOPE, LINE_SMOOTHING_FACTOR
    
    # Store original parameters
    original_canny_low = CANNY_LOW_THRESHOLD
    original_canny_high = CANNY_HIGH_THRESHOLD
    original_hough_threshold = HOUGH_THRESHOLD
    original_min_line_length = HOUGH_MIN_LINE_LENGTH
    original_max_line_gap = HOUGH_MAX_LINE_GAP
    original_min_slope = MIN_SLOPE
    original_smoothing_factor = LINE_SMOOTHING_FACTOR
    
    # Update with user-selected parameters
    
    CANNY_LOW_THRESHOLD = canny_low
    CANNY_HIGH_THRESHOLD = canny_high
    HOUGH_THRESHOLD = hough_threshold
    HOUGH_MIN_LINE_LENGTH = hough_min_line_length
    HOUGH_MAX_LINE_GAP = hough_max_line_gap
    MIN_SLOPE = min_slope
    LINE_SMOOTHING_FACTOR = smoothing_factor
    
    # Process the image
    result = process_image(image)
    
    # Restore original parameters
    CANNY_LOW_THRESHOLD = original_canny_low
    CANNY_HIGH_THRESHOLD = original_canny_high
    HOUGH_THRESHOLD = original_hough_threshold
    HOUGH_MIN_LINE_LENGTH = original_min_line_length
    HOUGH_MAX_LINE_GAP = original_max_line_gap
    MIN_SLOPE = original_min_slope
    LINE_SMOOTHING_FACTOR = original_smoothing_factor
    
    return result

# Image processing logic
if image_option == "Select from Dataset":
    # Create a selectbox with image filenames
    selected_image = st.sidebar.selectbox("Select an image:", image_files)
    
    if selected_image:
        # Load and display the selected image
        image_path = os.path.join(dataset_path, selected_image)
        image = cv2.imread(image_path)
        
        if image is not None:
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image_rgb, use_column_width=True)
            
            # Process and display the image with lane detection
            processed_image = process_image_with_params(image)
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Detected Lane Lines")
                st.image(processed_image_rgb, use_column_width=True)
            
            # Display image information
            st.subheader("Image Information")
            st.write(f"Filename: {selected_image}")
            st.write(f"Dimensions: {image.shape[1]} x {image.shape[0]} pixels")
            st.write(f"Dataset: {dataset_type}")
        else:
            st.error("Failed to load the selected image.")

else:  # Upload Custom Image
    uploaded_file = st.sidebar.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is not None:
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image_rgb, use_column_width=True)
            
            # Process and display the image with lane detection
            processed_image = process_image_with_params(image)
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Detected Lane Lines")
                st.image(processed_image_rgb, use_column_width=True)
            
            # Display image information
            st.subheader("Image Information")
            st.write(f"Uploaded Image")
            st.write(f"Dimensions: {image.shape[1]} x {image.shape[0]} pixels")
        else:
            st.error("Failed to load the uploaded image.")

# Add information about the KITTI dataset
st.sidebar.markdown("---")
st.sidebar.subheader("About KITTI Dataset")
st.sidebar.info("""
**KITTI Road Dataset**

The KITTI Vision Benchmark Suite provides road/lane detection evaluation datasets 
with ground truth for training and testing. The dataset contains images of various 
road scenes captured in urban, rural, and highway environments.

Image naming convention:
- um_: urban marked (lanes clearly marked)
- umm_: urban multiple marked (multiple lanes marked)
- uu_: urban unmarked (no lane markings)
""")

# Run the app with: streamlit run app.py