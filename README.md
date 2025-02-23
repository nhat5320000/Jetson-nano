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

---

# Install Python 3.8 on Jetson Nano

To install Python 3.8, follow these steps:

### 1. Install Required Dependencies

```bash
sudo apt update
sudo apt upgrade
sudo apt install build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev libffi-dev libc6-dev
```

### 2. Download and Compile Python 3.8

```bash
wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tar.xz
```

Extract and build:

```bash
tar -xf Python-3.8.12.tar.xz
cd Python-3.8.12
./configure --enable-optimizations
make -j4
sudo make altinstall
python3.8 --version
```

### 3. Create a Virtual Environment

```bash
python3.8 -m venv myenv
source myenv/bin/activate
```

### 4. Install Ultralytics and Convert Model to TensorRT Engine

```bash
pip3 install ultralytics
/usr/src/tensorrt/bin/trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```

### 5. Link TensorRT Engine to Python 3.8 Virtual Environment

```bash
source myenv/bin/activate
```

**Note:** Running `.pt` models directly on Python 3.8 works but is not optimized.

---

# Install Jetson Nano CUDA and OpenCV

To install CUDA and OpenCV for Jetson Nano, follow these steps:

### 1. Install OpenCV 4.5.0 with JetPack 4

OpenCV is required for image processing and deep learning applications.

- [Installation script for OpenCV 4.5.0 with JetPack 4](https://github.com/AastaNV/JEP/blob/master/script/install_opencv4.5.0_Jetpack4.sh)

```bash
git clone https://github.com/AastaNV/JEP.git && cd JEP/script && ./install_opencv4.5.0_Jetpack4.sh
```

### 2. Enable Swapfile for Better Performance

This increases virtual memory, helping Jetson Nano run heavy workloads.

- [Swapfile installation guide](https://github.com/JetsonHacksNano/installSwapfile)

```bash
git clone https://github.com/JetsonHacksNano/installSwapfile.git
cd installSwapfile
./installSwapfile.sh
```

### 3. Build OpenCV 4.5.1 Manually

For customized OpenCV versions with additional optimizations.

- [Custom OpenCV build script](https://github.com/JetsonHacksNano/buildOpenCV)

```bash
git clone https://github.com/JetsonHacksNano/buildOpenCV.git
cd buildOpenCV
gedit buildOpenCV.sh  # Edit OPENCV_VERSION=4.5.1

./buildOpenCV.sh |& tee openCV_build.log
```

*The installation process will take approximately 2 hours.*

---

## Hello AI World

**Hello AI World** is a project using NVIDIA Jetson to implement an object detection model based on SSD-Mobilenet. This project covers environment setup, data collection, model training, exporting ONNX files, and performing object detection.

### 1. Clone Repository

```bash
git clone --recursive https://github.com/dusty-nv/jetson-inference.git
```

### 2. Installation

```bash
cd jetson-inference
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install
```

### 3. Documentation

- [Installation and usage guide](https://github.com/dusty-nv/jetson-inference)
- [SSD training with PyTorch](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md)

### 4. Start Training and Detection

#### Enter Training Directory using Docker

```bash
cd jetson-inference
docker/run.sh
cd python/training/detection/ssd
```

#### Collect Data from Camera

```bash
# Capture from USB Camera
camera-capture /dev/video0

# Capture from CSI Camera
camera-capture csi://0
```

#### Train the Model

```bash
python3 train_ssd.py --dataset-type=voc --data=data/<dataset_name> \
                     --model-dir=models/<model_name> --batch-size=2 \
                     --workers=1 --epochs=1
```

#### Export ONNX File

```bash
python3 onnx_export.py --model-dir=models/<model_name>
```

#### Perform Detection and Export Supplementary ONNX File

```bash
detectnet --model=models/tractors/ssd-mobilenet.onnx --labels=models/tractors/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video0

detectnet --model=models/reserve_cloth/ssd-mobilenet.onnx --labels=models/reserve_cloth/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes csi://0
```

#### Resume Training

```bash
python3 train_ssd.py --dataset-type=voc --data=data/tractors \
                     --model-dir=models/tractors --resume=models/tractors/ssd-mobilenet.pth \
                     --batch-size=2 --workers=1 --epochs=1
```

## Conclusion

After completing these steps, you will have an SSD-Mobilenet model trained on customized data and capable of running on a Jetson device for real-time object detection.

**Good luck! 🚀**

�

