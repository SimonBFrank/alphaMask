#!/bin/sh

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

IMG_DIR="raw/images/"
ANNOTATION_DIR="raw/annotations/"

PROCESSED_IMG_DIR="processed/images/"
PROCESSED_ANNOTATIONS_DIR="processed/annotations/"

SCRIPT_PATH="extract_data.py"

python $SCRIPT_PATH $IMG_DIR $ANNOTATION_DIR $PROCESSED_IMG_DIR $PROCESSED_ANNOTATIONS_DIR
