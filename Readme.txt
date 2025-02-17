sudo apt-get install chromium-browser
sudo apt-get install python3-setuptools
sudo apt update
sudo apt install python3-pip
pip3 install imutils
#pip3 install nanocamera
git clone https://github.com/thehapyone/NanoCamera
cd NanoCamera
sudo python3 setup.py install
#pip3 install --user pyqt5
sudo apt-get install python3-pyqt5
sudo apt-get install pyqt5-dev-tools
sudo apt-get install qttools5-dev-tools
qtchooser -run-tool=designer -qt=5
#pyuic5 -x filename.ui -o filename.py
sudo apt-get install mysql-server
sudo mysql_secure_installation
Y-> pass:mysql 
sudo mysql -u root -p 
pass: mysql
#show database;

CREATE USER 'vqbg'@'localhost' IDENTIFIED BY 'vqbg123!';
GRANT ALL PRIVILEGES ON *.* TO 'vqbg'@'localhost';
exit;
sudo mysql -u vqbg -p
pass: vqbg123!
CREATE DATABASE CAMERA_PAPER;
exit;
# Bảng dữ liệu phần mềm sẽ tạo ra khi khỏi động SETTING_DATA_2
# USE CAMERA_PAPER;
# SELECT * FROM SETTING_DATA_2;

pip3 install mysql-connector
#sudo snap install pycharm-community --edge --classic Jetson orin

sudo apt install libcanberra-gtk-module libcanberra-gtk3-module
sudo /opt/nvidia/jetson-io/jetson-io.py
pip3 install pymodbus
sudo apt install v4l-utils
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --all

export PATH=/usr/local/cuda/bin:$PATH
cat /usr/local/cuda/version.txt
nvcc --version
pip3 install pycuda
sudo apt-get install ibus-unikey

# AI install cuda open CV 
Orin nano
#1. tải file cài open cv cuda
git clone https://github.com/mdegans/nano_build_opencv

#2.điều chỉnh file .sh
cd nano_build_opencv 
gedit ./build_opencv.sh
# diều chỉnh file theo cấu hình jetson và save
        -D CUDA_ARCH_BIN=5.3
        -D CUDA_ARCH_PTX=
        -D CUDA_FAST_MATH=ON
        -D CUDNN_VERSION='8.2'
# xóa 3 cai đặt thư viện và cài lại như bên dưới:
sudo apt install -y libdc1394-dev
sudo apt install -y python3-dev
sudo apt install -y python3-numpy

# install opencv cuda phiên bản open cv cao hơn phiên bản trong máy
./build_opencv.sh 4.9.0

# check cuda :
sudo pip3 install -U jetson-stats
sudo jtop 
# CUDA YES --> OK
# install utrlysic
pip3 install ultralytics
#check YOLO:

python3 Test_Cuda.py 
#####################
import cv2
from ultralytics import YOLO

# Tải mô hình YOLO
model = YOLO("yolov8n.pt")

# Mở webcam
cap = cv2.VideoCapture(0)  # 0 là camera mặc định

while cap.isOpened():
    ret, frame = cap.read()  # Đọc frame từ webcam
    if not ret:
        break
    
    results = model(frame)  # Chạy YOLO
    annotated_frame = results[0].plot()  # Vẽ kết quả

    cv2.imshow("YOLO Detection", annotated_frame)  # Hiển thị

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()
########################
# chuyển mô hình Pt--> RT
yolo export model=yolov8n.pt format=engine device=0
python3 pt_rt.py

##############
import tensorrt as trt

TRT_LOGGER = trt.Logger()
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Load mô hình ONNX
with open("yolov8n.onnx", "rb") as f:
    if not parser.parse(f.read()):
        print("Failed to parse ONNX file")
        exit()

# Build TensorRT engine
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
engine = builder.build_serialized_network(network, config)

# Lưu lại mô hình TensorRT
with open("yolov8n.trt", "wb") as f:
    f.write(engine)
print("Đã lưu YOLOv8 TensorRT model: yolov8n.trt")
#################################
###Test model trt. 
python3 Test_Cuda_rt.py

#############
from ultralytics import YOLO

# Load mô hình từ file .engine
model = YOLO("yolov8n.engine", task="detect")

# Đọc ảnh
import cv2

cap = cv2.VideoCapture(0)  # 0 là camera mặc định

while cap.isOpened():
    ret, frame = cap.read()  # Đọc frame từ webcam
    if not ret:
        break
    
    results = model.predict(frame, imgsz=640)  # Chạy YOLO
    annotated_frame = results[0].plot()  # Vẽ kết quả

    cv2.imshow("YOLO Detection", annotated_frame)  # Hiển thị

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()
###################################
# Huấn luyên mô hình 
# Install labelme
pip3 install labelme
pip3 install labelme2yolo
labelme
# chuyển đổi ảnh anh sang json
# Tạo thư mục lưu ảnh, gán nhãn ví dụ dataset_labelme
#########################
import os
import json
import glob
import os

# Định nghĩa cấu trúc thư mục
folders = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/labels/train",
    "dataset/labels/val"
]

# Tạo thư mục nếu chưa tồn tại
for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Thư mục đã được tạo thành công!")
#################################
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   ├── val/
│       ├── img3.jpg
│       ├── img4.jpg
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   ├── val/
│       ├── img3.txt
│       ├── img4.txt
└── data.yaml
######################
# Chuyển json qua txt
###########################
import os
import json
import glob
import os

# Định nghĩa cấu trúc thư mục
folders = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/labels/train",
    "dataset/labels/val"
]

# Tạo thư mục nếu chưa tồn tại
for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Thư mục đã được tạo thành công!")


# Thư mục chứa file JSON từ Labelme
json_dir = "dataset/labels_json"  
# Thư mục lưu file TXT đã chuyển đổi
output_dir = "dataset/labels"

# Đảm bảo thư mục đầu ra tồn tại
os.makedirs(output_dir, exist_ok=True)

# Đọc danh sách file JSON
json_files = glob.glob(os.path.join(json_dir, "*.json"))

# Danh sách các class (đổi theo dữ liệu của bạn)
class_names = ["OK", "NG"]  # Sửa theo danh sách của bạn

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]
    
    # Tạo tên file .txt
    txt_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + ".txt")

    with open(txt_filename, "w") as txt_file:
        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]

            # Kiểm tra nếu nhãn có trong danh sách
            if label not in class_names:
                continue

            class_id = class_names.index(label)

            # Tính bounding box (xmin, ymin, xmax, ymax)
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)

            # Chuyển sang định dạng YOLO (x_center, y_center, width, height) với tọa độ chuẩn hóa
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            # Ghi vào file TXT
            txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("Chuyển đổi hoàn tất!")

###########################
# tạo file data.yamal
import yaml

# Định nghĩa đường dẫn dữ liệu
dataset_path = "dataset"
train_path = f"{dataset_path}/images/train"
val_path = f"{dataset_path}/images/val"

# Danh sách các lớp (chỉnh sửa theo dữ liệu của bạn)
class_names = ["NG", "OK"]  # Ví dụ: nếu có 3 lớp thì ["cat", "dog", "bird"]

# Tạo dictionary chứa thông tin dataset
data_config = {
    "train": train_path,
    "val": val_path,
    "nc": len(class_names),  # Số lượng classes
    "names": class_names
}

# Ghi vào file data.yaml
yaml_path = f"{dataset_path}/data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

print(f"File data.yaml đã được tạo tại {yaml_path}")
####################################

# train tạo file pt
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=16 device=0
# sau khi train model sẽ được lưu ở đây : 
runs/detect/train2/weights/best.pt 
# chuyển đổi sang engine : 
yolo export model=runs/detect/train4/weights/best.pt format=engine device=0

#chạy mẫu mô hình


###end Jetson orrin 

# install jetson nano cuda open cv 
https://github.com/AastaNV/JEP/blob/master/script/install_opencv4.5.0_Jetpack4.sh


 git clone https://github.com/JetsonHacksNano/installSwapfile.git
cd installSwapfile
./installSwapfile.sh

git clone https://github.com/JetsonHacksNano/buildOpenCV.git
cd buildOpenCV
gedit buildOpenCV 
#repair : OPENCV_VERSION=4.5.1
./buildOpenCV.sh |& tee openCV_build.log

# finish about 2hours
##Install python 3

Steps for python 3.8 installation:
sudo apt update
sudo apt upgrade
sudo apt install build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev libffi-dev libc6-dev

Download the Python source code for version 3.8 from the official Python website. You can use the following command to download it directly to your Jetson Nano:
	wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tar.xz

Extract the downloaded archive by running the following command:
tar -xf Python-3.8.12.tar.xz
cd Python-3.8.12

Configure the build process:
	./configure --enable-optimizations
	 
Build Python:
make -j4

Once the compilation is complete, you can install Python by running the following command:
	sudo make altinstall
	python3.8 --version

That's it! You have successfully installed Python 3.8 on your Jetson Nano.

Now come out from python3.8 folder and create a separate environment using python 3.8
python3.8 -m venv myenv                                                
source myenv_1/bin/activate

pip3 install ultralytics
#chuye pt sang engine
/usr/src/tensorrt/bin/trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
#lieen ket tensor voi moi truong ao 3.8
source myenv/bin/activate
## chyaj .pt tren 3.8 dduowcj nuwng ko toi uu 

# Huan luyen xong chuyen mohinhf va chya thu banwg deepstream
  

### Helol Ai world


# Clone repository
git clone --recursive https://github.com/dusty-nv/jetson-inference.git

# Cài đặt
cd jetson-inference
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install

# huowngs dan
https://github.com/dusty-nv/jetson-inference
https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md

##Start :
# vao duogn dan train banng docker/run.sh
cd jetson-inference
docker/run.sh
cd python/training/detection/ssd
ls
# vidu download anh ve va train
python3 open_images_downloader.py --class-names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon" --data=data/fruit

#python3 train_ssd.py --data=data/fruit --model-dir=models/fruit --batch-size=4 --epochs=30

python3 train_ssd.py --data=data/fruit --model-dir=models/fruit --batch-size=2 --workers=1 --epochs=1
python3 onnx_export.py --model-dir=models/fruit


detectnet --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes "/jetson-inference/data/images/fruit_*.jpg" /jetson-inference/data/images/fruit_%i.jpg

# tu train            
camera-capture /dev/video0
camera-capture csi://0
# gan anhr 
python3 train_ssd.py --dataset-type=voc --data=data/tractors --model-dir=models/tractors --batch-size=2 --workers=1 --epochs=1
python3 train_ssd.py --dataset-type=voc --data=data/reserve_cloth --model-dir=models/reserve_cloth --batch-size=2 --workers=1 --epochs=1
python3 onnx_export.py --model-dir=models/tractors
python3 onnx_export.py --model-dir=models/reserve_cloth

detectnet --model=models/tractors/ssd-mobilenet.onnx --labels=models/tractors/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video0

detectnet --model=models/reserve_cloth/ssd-mobilenet.onnx --labels=models/reserve_cloth/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes csi://0


END

