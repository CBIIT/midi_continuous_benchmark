#! /bin/bash

echo "----------------- 1: DICOM validation ------------------"
python3 run_validation.py config_example_linux.json

echo "----------------- 2: Scoring ------------------"
python3 run_reports.py config_example_linux.json

echo "----------------- 3: DICOM verification ------------------"
python3 run_dciodvfy.py config_example_linux.json

echo "----------------- 4: Image comparison ------------------"
python3 run_DICOMImgCmp.py config_example_linux.json
