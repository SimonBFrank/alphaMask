rm -r ./alphaMask

mkdir ./alphaMask
mkdir ./alphaMask/images
mkdir ./alphaMask/images/training
mkdir ./alphaMask/images/validation
mkdir ./alphaMask/labels
mkdir ./alphaMask/labels/training
mkdir ./alphaMask/labels/validation

python export_data.py

cp -r ./alphaMask/ ./../../yolov5/data

rm -r ./__pycache__/