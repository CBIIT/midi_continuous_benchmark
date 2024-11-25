## Objective

The script is dedicated to the project of MIDI Continuous Benchmark. The main validation script is derived from the Validation Script repo at https://github.com/CBIIT/MIDI_validation_script.git (version 3.14)

## Requirement 

To execute the script, a configuration file is required. The configuration file must contain the following information, please refer to the template.
### Configuration file
| Configure             | Description|
|-----------------------|------------|
|"run_name"             | job name|
|"input_data_path"      | path to the folder containing the de-identified images|
|"output_data_path"     | path to the output|
|"answer_db_file"       | path to the Answerkey, which contains all correct answers|
|"uid_mapping_file"     | path to the UID mapping file|
|"patid_mapping_file    | path to the patient ID mapping file|
|"multiprocessing"      | whether multiprocessing. Default: True|
| "multiprocessing_cpus"| number of CPUs for the multiprocessing. Default: 5|
| "log_path"            | path to the logs|
| "log_level"           | log level, Default: info |
| "report_series"       | whether series level based or instance level based. Default: False|
| "pre_deID_data_path" 	| path to the source data (pre de-identification)|

## Workflow
The complete workflow consists of four parts (assuming in a Linux system):

1. DICOM validation - to validate the DICOM de-identification
   ```python 
   python run_validation.py config_example_linux.json
   ```

2. Scoring - to compute the accuracy in percentage according to the validation results
   ```python
   python run_reports.py config_example_linux.json
   ```

3. DICOM verification - to validate the DICOM compliance
   ```python
   python run_dciodvfy.py config_example_linux.json
   ```

4. Image comparison - to compare the pixel image between before and after de-identification and the result stored in a pdf file
   ```python
   python run_DICOMImgCmp.py config_example_linux.json
   ```

To simplify the pipeline, the main.sh combines all four parts in one batch script. 
```bash
bash main.sh
```
