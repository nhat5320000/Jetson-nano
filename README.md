# Jetson Nano Setup Repository

This repository contains scripts and resources to set up a Jetson Nano for camera and database applications.

## Contents
- `setup_jetson.sh`: Shell script to install required packages and configure the system.
- `camera_script.py`: Python script for camera operations.
- `database_utils.py`: Python script for MySQL database interactions.
- `Images`: Diagrams and setup images.

## Usage

### 1. Download the Basic PyQT5 Interface

1. Clone this repository:
    ```bash
    git clone https://github.com/nhat5320000/Jetson-nano.git
    ```

2. Run the setup script:
    ```bash
    cd Jetson-nano
    chmod +x setup_jetson.sh
    sudo ./setup_jetson.sh
    ```

### 2. Download Object Detection (SSD-Mobilenet)

1. Clone this repository:
    ```bash
    git clone https://github.com/nhat5320000/Jetson-nano.git
    ```

2. Run the setup script:
    ```bash
    cd Jetson-nano
    chmod +x install_jetson_inference.sh
    sudo ./install_jetson_inference.sh
    ```
