# Principles of Robot Autonomy I HW4
Homework 4 code for Principles of Robot Autonomy I
## Dependencies
Python package dependencies for this assignment are listed in the `requirements.txt` file and can be automatically installed by
```
pip install -r requirements.txt
```
Problem 2 also needs GTSAM, installed separately with `pip install gtsam` (Python 3.11 or newer). GTSAM has no pip package for Windows; on Windows, use `conda install -c conda-forge gtsam`, your Ubuntu VM, or Google Colab (the first cell of `p2_slam.ipynb` clones this repo and installs GTSAM; edit `P2_slam.py` in the Colab file editor and download it before the runtime resets). Problem 1 does not need GTSAM.
## Submission
Upload `P1_features.py` and `P2_slam.py` to the HW4 Code assignment on Gradescope. Do not rename the files or change the function signatures. You do not need to change or submit `utils.py`.
