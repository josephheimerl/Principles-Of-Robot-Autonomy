# Principles of Robot Autonomy I HW3
Homework 3 code for Principles of Robot Autonomy I
## Dependencies
Python package dependencies for this assignment are listed in the `requirements.txt` file and can be automatically installed by
```
pip install -r requirements.txt
```
OpenCV is only used to detect the checkerboard corners in Problem 1. If it does not install on your machine, the notebook falls back to the corners saved in `data/corners.npz`.
Problem 4 (iv) also needs PyTorch: `pip install torch torchvision` (the CPU version is fine), or run that notebook in Google Colab.
## Submission
Upload `P1_extrinsics.py`, `P2_ransac.py`, `P3_icp.py`, and `P4_detection.py` to the HW3 Code assignment on Gradescope. Do not rename the files or change the function signatures.
