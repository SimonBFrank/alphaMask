python select_IMFD.py
python select_AMFD.py

IMG_DIRS="../Kaggle_data/images/,../imfd_selected/imgs/,../amfd_selected/imgs/"
ANNOTATION_DIRS="../Kaggle_data/annotations/,../imfd_selected/labels/,../amfd_selected/labels/"

PROCESSED_IMG_DIR="../processed/images/"
PROCESSED_ANNOTATIONS_DIR="../processed/annotations/"

SCRIPT_PATH="extract_data.py"

python $SCRIPT_PATH $IMG_DIRS $ANNOTATION_DIRS $PROCESSED_IMG_DIR $PROCESSED_ANNOTATIONS_DIR

rm -r ./data

mkdir ./data
mkdir ./data/images
mkdir ./data/images/training
mkdir ./data/images/validation
mkdir ./data/labels
mkdir ./data/labels/training
mkdir ./data/labels/validation

python export_data.py

rm -r ./__pycache__/