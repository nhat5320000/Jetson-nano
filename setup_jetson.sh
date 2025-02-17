#!/bin/bash

# Cập nhật hệ thống và cài đặt package cơ bản
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y chromium-browser python3-setuptools python3-pip
sudo apt-get install -y python3-pyqt5 pyqt5-dev-tools qttools5-dev-tools
sudo apt-get install -y libcanberra-gtk-module libcanberra-gtk3-module
sudo apt-get install -y v4l-utils ibus-unikey

# Cài đặt các thư viện Python
pip3 install imutils pymodbus mysql-connector-python pycuda

# Cài đặt NanoCamera
git clone https://github.com/thehapyone/NanoCamera
cd NanoCamera || exit
sudo python3 setup.py install
cd ..

# Cấu hình CUDA
echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc

# Cài đặt và cấu hình MySQL
sudo apt-get install -y mysql-server

# Tự động cấu hình MySQL (password: mysql)
sudo mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'mysql'; FLUSH PRIVILEGES;"

# Tạo user và database
sudo mysql -u root -pmysql -e "CREATE USER 'vqbg'@'localhost' IDENTIFIED BY 'vqbg123!'; GRANT ALL PRIVILEGES ON *.* TO 'vqbg'@'localhost'; CREATE DATABASE CAMERA_PAPER;"

echo "Cài đặt hoàn tất!"
