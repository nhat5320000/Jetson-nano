# Jetson Nano Setup Repository

This repository contains scripts and resources to set up a Jetson Nano for camera and database applications.

## Contents
- `setup_jetson.sh`: Shell script to install required packages and configure the system.
- `camera_script.py`: Python script for camera operations.
- `database_utils.py`: Python script for MySQL database interactions.
- `Images`: Diagrams and setup images.

## Usage for download the basic PyQT5 interface 
Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git

1.Usage for download the basic PyQT5 interface
   Run the setup script:
   ```bash
   cd your-repo-name
   chmod +x setup_jetson.sh
   sudo ./setup_jetson.sh

2.Usage for download Object Detection (SSD-Mobilenet)
   Run the setup script:
   ```bash
   cd your-repo-name
   chmod +x install_jetson_inference.sh
   sudo ./install_jetson_inference.sh
