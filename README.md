# LaneLineDetection

A computer vision project for detecting lane lines in road images using OpenCV and Python.

## Project Structure

- `main.py` - Core lane detection logic and algorithms
- `config.py` - Configuration parameters for tuning the detection
- `gui.py` - Tkinter-based GUI implementation
- `streamlit_app.py` - Interactive Streamlit dashboard
- `app.py` - Streamlit web application for lane detection visualization
- `requirements.txt` - Required dependencies

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The project offers multiple interfaces:

### Streamlit Web Application

Run the Streamlit app for an interactive web interface:

```bash
streamlit run app.py
```

This provides a user-friendly interface with parameter tuning and visualization.

## Features

- Lane detection in images from the KITTI dataset
- Support for custom image uploads
- Interactive parameter tuning
- Real-time visualization of detection results
- Color filtering for white and yellow lanes
- Edge detection using Canny algorithm
- Region of interest selection
- Hough line detection
- Line averaging and drawing

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Streamlit
- MoviePy

## Dataset

This project uses the KITTI Road Dataset, which provides road/lane detection evaluation datasets with ground truth for training and testing.#   L a n e L i n e D e t e c t i o n  
 