import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt
from config import *
from main import (
    process_image,
    interested_region,
    hough_lines,
    weighted_img,
    lines_drawn
)

# Set page configuration
st.set_page_config(page_title="Lane Line Detection", layout="wide")

# Title and description
st.title("🚗 Lane Line Detection System using OpenCV")

st.markdown("""
## About this Project
This project demonstrates a real-time Lane Line Detection pipeline using Computer Vision.
It mimics the kind of technology used in ADAS (Advanced Driver Assistance Systems).

The pipeline consists of several steps:
1. Color filtering (white and yellow lanes)
2. Edge detection using Canny algorithm
3. Region of interest selection
4. Hough line detection
5. Line averaging and drawing
""")

# Sidebar for parameters
st.sidebar.header("Parameters")

# Sample videos or upload option
st.sidebar.subheader("Video Input")
video_option = st.sidebar.radio(
    "Choose video source:",
    ("Upload Video", "Use Sample Video")
)

# Parameter tuning in sidebar
st.sidebar.subheader("Parameter Tuning")

# Canny Edge Detection Parameters
st.sidebar.markdown("### Edge Detection")
canny_low = st.sidebar.slider("Canny Low Threshold", 10, 100, CANNY_LOW_THRESHOLD)
canny_high = st.sidebar.slider("Canny High Threshold", 50, 200, CANNY_HIGH_THRESHOLD)

# Hough Transform Parameters
st.sidebar.markdown("### Hough Transform")
hough_threshold = st.sidebar.slider("Hough Threshold", 10, 50, HOUGH_THRESHOLD)
hough_min_line_length = st.sidebar.slider("Min Line Length", 50, 150, HOUGH_MIN_LINE_LENGTH)
hough_max_line_gap = st.sidebar.slider("Max Line Gap", 50, 250, HOUGH_MAX_LINE_GAP)

# Line Parameters
st.sidebar.markdown("### Line Detection")
min_slope = st.sidebar.slider("Minimum Slope", 0.1, 1.0, MIN_SLOPE)
smoothing_factor = st.sidebar.slider("Smoothing Factor", 0.1, 1.0, LINE_SMOOTHING_FACTOR)

# Function to process a single frame with visualization of each step
def process_frame_with_steps(frame):
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
    
    # Process the frame
    result = process_image(frame)
    
    # Restore original parameters
    CANNY_LOW_THRESHOLD = original_canny_low
    CANNY_HIGH_THRESHOLD = original_canny_high
    HOUGH_THRESHOLD = original_hough_threshold
    HOUGH_MIN_LINE_LENGTH = original_min_line_length
    HOUGH_MAX_LINE_GAP = original_max_line_gap
    MIN_SLOPE = original_min_slope
    LINE_SMOOTHING_FACTOR = original_smoothing_factor
    
    return result

# Function to generate intermediate steps for visualization
def generate_step_images(frame):
    # Step 1: Convert to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply color thresholds
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(img_hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH)
    mask_white = cv2.inRange(gray_image, WHITE_THRESHOLD_LOW, WHITE_THRESHOLD_HIGH)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
    
    # Step 3: Apply Gaussian blur
    gauss_gray = cv2.GaussianBlur(mask_yw_image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    
    # Step 4: Apply Canny edge detection
    canny_edges = cv2.Canny(gauss_gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    
    # Step 5: Apply region of interest
    vertices = [get_roi_vertices(frame.shape)]
    roi_image = interested_region(canny_edges, vertices)
    
    # Step 6: Apply Hough transform
    line_image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    lines = cv2.HoughLinesP(roi_image, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, 
                          np.array([]), minLineLength=HOUGH_MIN_LINE_LENGTH, 
                          maxLineGap=HOUGH_MAX_LINE_GAP)
    
    # Create a copy for visualization
    hough_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Step 7: Draw lane lines
    lane_lines_image = np.zeros_like(frame)
    if lines is not None:
        lines_drawn(lane_lines_image, lines)
    
    # Step 8: Blend with original image
    final_image = weighted_img(lane_lines_image, frame)
    
    # Convert all images to RGB for display
    gray_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    mask_yw_rgb = cv2.cvtColor(mask_yw_image, cv2.COLOR_GRAY2RGB)
    canny_rgb = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB)
    roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2RGB)
    
    return {
        "original": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        "grayscale": gray_rgb,
        "color_mask": mask_yw_rgb,
        "canny": canny_rgb,
        "roi": roi_rgb,
        "hough": cv2.cvtColor(hough_image, cv2.COLOR_BGR2RGB),
        "lane_lines": cv2.cvtColor(lane_lines_image, cv2.COLOR_BGR2RGB),
        "final": cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    }

# Main content area with tabs
tab1, tab2 = st.tabs(["Video Processing", "Pipeline Explanation"])

with tab1:
    # Video processing tab
    if video_option == "Upload Video":
        uploaded_video = st.file_uploader("Upload a Road Video", type=['mp4', 'mov', 'avi'])
        video_path = None
        
        if uploaded_video is not None:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
    else:
        # Use sample video
        video_path = "test2.mp4"  # Assuming this file exists in the project directory
        st.info(f"Using sample video: {video_path}")
    
    if video_path:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Error opening video file")
        else:
            # Video controls
            st.subheader("Video Controls")
            col1, col2 = st.columns(2)
            
            with col1:
                play_button = st.button("Play")
            
            with col2:
                stop_button = st.button("Stop")
            
            # Display area for video
            st.subheader("Lane Detection Output")
            video_placeholder = st.empty()
            
            # Process and display video frames
            if play_button:
                stop_processing = False
                while cap.isOpened() and not stop_processing:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize frame for display
                    frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                    
                    # Process frame with user-selected parameters
                    processed_frame = process_frame_with_steps(frame)
                    
                    # Display the processed frame
                    video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                    
                    # Check if stop button was pressed
                    if stop_button or not play_button:
                        stop_processing = True
                        break
            
            cap.release()

with tab2:
    # Pipeline explanation tab
    st.subheader("Lane Detection Pipeline Visualization")
    st.markdown("""
    This tab shows each step of the lane detection pipeline on a single frame.
    Adjust the parameters in the sidebar to see how they affect each step.
    """)
    
    # Use a sample frame for demonstration
    sample_frame = None
    
    if video_path:
        cap = cv2.VideoCapture(video_path)
        ret, sample_frame = cap.read()
        cap.release()
        
        if ret:
            sample_frame = cv2.resize(sample_frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
    
    if sample_frame is not None:
        # Generate all step images
        step_images = generate_step_images(sample_frame)
        
        # Display steps in a grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 1. Original Image")
            st.image(step_images["original"], use_column_width=True)
            
            st.markdown("### 3. Color Thresholding")
            st.image(step_images["color_mask"], use_column_width=True)
            
            st.markdown("### 5. Region of Interest")
            st.image(step_images["roi"], use_column_width=True)
            
            st.markdown("### 7. Lane Lines")
            st.image(step_images["lane_lines"], use_column_width=True)
        
        with col2:
            st.markdown("### 2. Grayscale Conversion")
            st.image(step_images["grayscale"], use_column_width=True)
            
            st.markdown("### 4. Canny Edge Detection")
            st.image(step_images["canny"], use_column_width=True)
            
            st.markdown("### 6. Hough Transform")
            st.image(step_images["hough"], use_column_width=True)
            
            st.markdown("### 8. Final Result")
            st.image(step_images["final"], use_column_width=True)
        
        # Explanation of each step
        with st.expander("Detailed Explanation of Each Step"):
            st.markdown("""
            1. **Original Image**: The input frame from the video.
            
            2. **Grayscale Conversion**: Converting the image to grayscale simplifies processing.
            
            3. **Color Thresholding**: Isolating white and yellow colors to detect lane markings.
            
            4. **Canny Edge Detection**: Identifying edges in the image using gradient changes.
            
            5. **Region of Interest**: Focusing only on the road area where lanes are expected.
            
            6. **Hough Transform**: Detecting line segments from the edges.
            
            7. **Lane Lines**: Averaging and smoothing the detected lines to form continuous lane markings.
            
            8. **Final Result**: Overlaying the detected lanes on the original image.
            """)

# Footer
st.markdown("---")
st.markdown("""
### About the Project
This Lane Line Detection system uses computer vision techniques to identify and track lane markings in road videos.
It's built using OpenCV and demonstrates fundamental concepts in autonomous driving technology.

### Future Improvements
- Improved curve detection
- Better handling of shadows and lighting changes
- Lane departure warning system
- Distance estimation to lane boundaries
""")