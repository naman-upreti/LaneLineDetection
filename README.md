
# Lane Line Detection 🚗🛣️  
*A Computer Vision Project for Detecting Lane Lines in Road Images using OpenCV & Python.*

---

## Table of Contents

- [Overview](#overview)  
- [Project Structure](#project-structure)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Streamlit Web Application](#streamlit-web-application)  
- [Technologies Used](#technologies-used)  
- [Dataset](#dataset)  
- [Screenshots](#screenshots)  
- [License](#license)

---

## Overview

This project focuses on detecting lane lines from road images using Computer Vision techniques with OpenCV and Python.

It provides a real-time visualization interface with customizable parameters and multiple interaction modes via GUI (Tkinter) or Web Application (Streamlit).

---

## Project Structure

```
LaneLineDetection/
│
├── main.py              # Core lane detection logic and algorithms  
├── config.py            # Configuration parameters for tuning detection  
├── gui.py               # Tkinter-based GUI Implementation  
├── streamlit_app.py     # Streamlit dashboard (alternative app interface)  
├── app.py               # Primary Streamlit web application  
├── requirements.txt     # Required dependencies  
└── README.md            # Project documentation  
```

---

## Features 🚀

- Lane detection in images from the KITTI dataset  
- Real-time visualization of detection results  
- Support for custom image uploads  
- Color filtering (White & Yellow lanes)  
- Edge detection using Canny Algorithm  
- Region of Interest (ROI) Selection  
- Hough Line Transform for line detection  
- Line averaging & lane drawing  
- Interactive parameter tuning through GUI or Web App  

---

## Installation ⚙️

Clone the repository:

```bash
git clone https://github.com/your-username/LaneLineDetection.git
cd LaneLineDetection
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage 🎯

### Streamlit Web Application (Recommended)

Launch the interactive web interface:

```bash
streamlit run app.py
```

This will start a local Streamlit server where you can:

- Upload custom road images  
- Adjust detection parameters  
- View real-time lane detection visualization  

---

## Technologies Used 🛠️

- Python  
- OpenCV  
- NumPy  
- Matplotlib  
- Streamlit  
- Tkinter  
- MoviePy  

---

## Dataset 📂

This project utilizes the [KITTI Road Dataset](http://www.cvlibs.net/datasets/kitti/) for lane detection tasks.

> The KITTI dataset provides high-quality road and lane detection evaluation datasets with ground truth labels for training and testing.

---

## Screenshots 📸

> Add relevant screenshots here for better visualization  
> (Optional but highly recommended)

---

## License 📄

This project is open-source and available under the MIT License.

---
