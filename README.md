# YoloV2 Inference Engine
Projects for __CS492 Systems for Machine Learning__ course, Spring 2020, KAIST

## About
This repository includes implementations of essential layers to infer YOLOv2 model. Layers has been implemented in using only python `numpy` API as well as using CPU and GPU parallelization techniques.

## Table of Contents:
+ data
  - input for the model
+ src
  - Contains implementation of projects
  - The skeleton for `yolov2tiny.py` and `__init__.py` was provided from the course.
+ models
  - pickle file that contains pretrained YOLOv2 model's weights
+ docs
  - Includes project details and project reports

## Projects

### proj1
Runs inference of `yolov2tiny` model using existing library (`tensorflow` API).

### proj2
Implementation of YoloV2 Inference Engine using only python `numpy` API.

### proj3
Implementation of YoloV2 Inference Engine using various parallelization techniques.
- CPU parallelization
    - High level library: `OpenBLAS`
    - Manual: `AVX`, `pthread`
- GPU parallelization
    - High level library: `cuBLAS`
    - Manual: `CUDA`
