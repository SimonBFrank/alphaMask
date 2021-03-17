python select_IMFD.py

IMG_DIRS="../Kaggle_data/images/,../imfd_selected/imgs/"
ANNOTATION_DIRS="../Kaggle_data/annotations/,../imfd_selected/labels/"

PROCESSED_IMG_DIR="../processed/images/"
PROCESSED_ANNOTATIONS_DIR="../processed/annotations/"

SCRIPT_PATH="extract_data.py"

python $SCRIPT_PATH $IMG_DIRS $ANNOTATION_DIRS $PROCESSED_IMG_DIR $PROCESSED_ANNOTATIONS_DIR

rm -r ./alphaMask

mkdir ./alphaMask
mkdir ./alphaMask/images
mkdir ./alphaMask/images/training
mkdir ./alphaMask/images/validation
mkdir ./alphaMask/labels
mkdir ./alphaMask/labels/training
mkdir ./alphaMask/labels/validation

python export_data.py

rm -r ./__pycache__/