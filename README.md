# Gaze4HRI: A Large-scale Dataset for Gaze Estimation in Human-Robot Interaction (FG 2026)

This repository contains the main codebase for the "Gaze4HRI: Zero-shot Benchmarking Gaze Estimation Neural-Networks for Human-Robot Interaction" paper accepted to the 20th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2026). It includes scripts for data collection, dataset creation (HDF5), gaze/blink estimation, and the analysis suites used to generate the results presented in the paper.

## Content
1. Dataset Overview
2. Testing a Gaze Model (How we tested and analyzed each gaze model, step by step)
3. Script Reference (To explain what each script in the codebase does)

## 1. Dataset Overview

### Raw Data Content

Each of the four main setups used in the dataset (shown in "Fig. 4: The four setups used in our analysis." of the paper) contains one more sub-types, which are referred to as experiment types (or shortly "exp_type") in the raw dataset. As can be seen on the config.py script, this is the list of all experiment types:

["lighting_10", "lighting_25", "lighting_50", "lighting_100", "head_pose_left", "head_pose_middle", "head_pose_right", "circular_movement", "line_movement_slow", "line_movement_fast"]

* lighting_xxx includes recording at each illumination level in the (a) Illumination Setup.
* circular_movement is from the (b) Camera Viewpoint Setup
* head_pose_left/middle/right are from the (c) Head-Gaze Conflict Setup.
* line_movement_slow/fast are from the (d) Moving Target (Mutual Gaze) Setup.

Each of these exp_type directories have a set of points as subdirectories as explained in the next section. Please refer to config.py for which set of points are found under each exp_type.

### Raw File Structure

Except for training or `h5_` prefixed scripts, the raw format of the Gaze4HRI dataset is required for almost all scripts, as this was the file structure used during the writing of the paper. The raw file structure is organized in a nested directory structure as follows:

* **Subject Directory (`subject_dir`):** `YYYY-MM-DD/SubjectName`

* **Experiment Directory (`exp_dir`):** `YYYY-MM-DD/SubjectName/exp_type/point`

**Note on Timestamps:** Since some videos required re-recording, each `exp_dir` contains subdirectories named with a timestamp. During evaluation, we always use the directory with the latest timestamp: `.../exp_type/point/latest_timestamp` by setting `get_latest_subdirectory_by_name=True` in the `GazeDataLoader` constructor.

### Dependencies

We used a branch of the `GazeModels` repository for each gaze model we tested in the paper.

## 2. Testing a Gaze Model
Although it is not the most optimal way, the following was how we analyzed gaze models during the writing of the paper. Note that if you want to analyze multiple gaze models, you need to do steps 1 and 2 for each model. Then, you can execute steps 3 and onwards only once to analyze multiple gaze models together.
1. **Do inference (gaze estimation) with a gaze model** by using the corresponding branch in the GazeModels repository.
2. **Import/Export**: If your GazeModel (for inference) and Gaze4HRI (for analysis) repos are on different machines (which was the case for us), you should export/import gaze estimations as follows. As explained in the correspdoning branch's `READMEGazeModel.md`, the gaze estimations should be exported by running `flatten_gaze_estimations` within `flatten_dir.py`. Then, in this main Gaze4HRI repo, you should run `unflatten` within `flatten_dir.py` to import the gaze estimations.
3. **Inverse Data Rectification:**  The raw Gaze4HRI dataset's ground-truth gaze vectors are not rectified, but some gaze models (e.g. ETH-X-Gaze trained PureGaze) predict rectified gaze vectors. For such models, their gaze estimations were transformed (with the inverse data rectification transformation) via `unrectification.py` to match the raw Gaze4HRI dataset's ground-truth gaze vectors. So, for such models only, use `unrectification.py`.
4. **Gaze Error Calculation and Aggregation**
* Set `.env/CURRENTLY_ANALYZED_MODELS=gaze_model1,gaze_model2...` to the list of the models you want to analyze.
* `python data_analyzer_batch.py` to calculate gaze error on each video for each model.
* `python structured_results.py` to produce `gaze_evaluation_results.csv` to be used in subsequent analysis scripts.
5. **Analysis Scripts**:
We finally ran analysis scripts such as those listed in "Paper-Specific Scripts" of the "3. Script Reference" section of this README.

## 3. Script Reference

### Core Data Management & Utilities

* `config.py`: The essential file that encompasses all widely-used helpers for conducting data collection and analysis.

* `data_loader.py`: Contains the `GazeDataLoader` class used to load video, head poses, and gaze ground-truth of a recorded `exp_dir`.

* `data_matcher.py`: Performs sensor fusion and synchronization between the 100Hz Motion Capture data and the 30Hz image data.

* `flatten_dir.py`: Essential script to export/import into the nested structure of the raw Gaze4HRI dataset. It flattens leaf files for export or unflattens them for import.

* `frame_db.py`: A parquet database script used indirectly across various analyses.

* `generate_subject_id_mapping.py`: Used to map actual subject names to subject IDs to preserve privacy.

* `subject_stats.py`: Creates `subject_stats.json` from `subject_metadata.json`.

### Gaze4HRI Torch Dataset (Gaze Estimation)
These scripts can be used to train gaze models on Gaze4HRI. Uses rectified images as input.

* `h5_dataset.py`: The PyTorch dataset class for Gaze4HRI.

* `h5_dataset_creator.py`: Creates the HDF5 files for the Gaze4HRI torch dataset.

* `create_linear_dataset.py`: Helper script used to debug before creating the full torch dataset; useful if modified dataset generation is required.

* `data_rectification.py`: Performs data rectification using the literature-standard code provided by Gazehub.

* `raw_to_ethxgaze.py`: Converts raw Gaze4HRI data (non-rectified) to the ethxgaze format (rectified).

* `ethxgaze_to_raw.py`: Converts rectified ethxgaze data back to the raw Gaze4HRI format.

* `unrectification.py`: Converts gaze estimations from rectified format back to the raw Gaze4HRI format for paper analysis.

### Blink4HRI Torch Dataset (Blink Detection)
These scripts can be used to train blink models on Blink4HRI. Uses exordium_landmarks as used in the BlinkLinMulT (Fodor et al., 2023) study.

* `h5_blink4hri.py`: The PyTorch dataset class for Blink4HRI.

* `h5_blink4hri_creator.py`: Creates the HDF5 files for the Blink4HRI torch dataset.

* `h5_blink4hri_split.py`: Splits the Blink4HRI dataset into train/val/test sets while maintaining dataset balance.

* `h5_blink4hri_stats.py`: Computes descriptive statistics for the Blink4HRI dataset.

* `exordium_landmarks.py`: Adapted from the official BlinkLinMulT repo. Creates face/eye crops and eye aspect ratio (EAR) features.

* `exordium_landmarks_batch.py`: Processes the entire dataset to generate exordium landmarks.

* `generate_feature_stats.py`: Used to normalize exordium_landmarks features for the Blink4HRI dataset.

* `talkingface.json`: Used for testing default BLMT weights (from official repo).

### Inference & Estimation
These were the inference/test scripts used during the writing of the paper.

* **Gaze Estimation:**

  * `gaze_estimation.py`: Contains the `GazeModel` abstract class used to test any gaze model on the torch dataset.

  * `gaze_estimation_batch.py`: Runs the `gaze_estimation.py` logic across all `exp_dirs`.

  * `gaze_model_mcgaze.py`: Implementation of the MCGaze class. Unlike other models (which reside in the `GazeModels` repo), this script is kept here as it was used for real-time testing. It uses **UNIX Domain Sockets** to communicate with the MCGaze model environment.

  * `head_detection.py` / `head_detection_batch.py`: Detects face bounding boxes to determine cropping configurations for gaze model inputs.

* **Blink Estimation:**

  *  `blink_annotation.py`: Used to do manual blink annotations on raw video. Used for masking blink frames in Gaze4HRI evaluation. Serves as blink ground-truth in Blink4HRI.
  * `blmt_train.py` / `blmt_test.py`: Scripts to train and test **BlinkLinMulT (BLMT)** (Fodor et al., 2023) on the Blink4HRI splits.

### Analysis & Paper Experiments
These were the analysis scripts used during the writing of the paper.

* **Evaluation Pipeline:**

  * `data_analyzer.py`: Key analysis script that calculates the angular gaze error for each frame of a video.

  * `data_analyzer_batch.py`: Runs `data_analyzer.py` on all experiment directories.

  * `structured_results.py`: Reads gaze estimation error files from each `exp_dir` and aggregates them into a single `gaze_evaluation_results.csv`.

  * `analyze_by_group.py`: Analyzes gaze errors by aggregating by model, experiment type, target point, subject gender, etc. (Requires `structured_results.py` output).

  * `blink_structured_results.py` / `blink_analyze_by_group.py`: Identical to the gaze analysis pipeline but applied to blink estimation results.

* **Paper-Specific Scripts:**

  * `lighting_analysis.py`: Generates results for **"Exp. 1: Illumination"**.

  * `camera_viewpoint.py`: Generates results for **"Exp. 2: Camera Viewpoint"**.

  * `head_gaze_conflict_vs_error.py`: Generates results for **"Experiment 3: Head–gaze conflict"**.

  * `point_analysis.py`: Generates results for **"Exp. 4.1. Object-Centered Setup"**.

  * `line_movement_analysis.py` / `line_movement_analysis_paper.py`: Generates results for **"Exp. 4.2. Mutual-Gaze Setup"**.

  * `crop_vs_rect.py`: Analysis for **"Table S2: Crop vs Rectification"** in the Supplementary Material.

  * `pitch_yaw_stats.py`: Creates statistics for head-forward and gaze pitch-yaw values, used in **"TABLE S4: Exp. 4.1"**.

  * `dataset_stats.py`: Calculates the total number of frames and state statistics for the entire dataset.

  * `neutral_cam_position_calculation.py` / `neutral_eye_position_calculation.py`: Calculates mean/median front-facing poses to create `neutral_eye_position_per_subject.csv`.

### Data Collection & Calibration
These were the data collection scripts used during the recording of the dataset.


* `data_collector.py`: The main script used to record the dataset during data collection.

* `eye_broadcaster.py`: Publishes eye positions to ROS during data collection; not used in post-hoc analysis.

* `data_visualizer.py`: Visualizes gaze ground-truth and estimations on **RVIZ** for verification during both collection and analysis stages.

* `head_eye_calibration.py`: Used for the head-eye calibration procedure described in the paper.

* `table_target_calibration.py`: Calibrates the static transformation for "Table to Gaze Target Point" (performed once per target point).

* `sounds/`: Contains audio files (e.g., "Start", "Finish") used to instruct subjects during recording.