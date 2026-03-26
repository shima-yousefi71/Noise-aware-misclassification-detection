# Noise-Aware Misclassification Attack Detection in Collaborative DNN Inference

## Overview
This repository provides the implementation of a noise-aware framework for detecting adversarial misclassification attacks in collaborative DNN inference.

The method operates on intermediate features extracted from a VGG19 model (layer 20) trained on CIFAR-100. It considers both adversarial manipulation and noise during detection.
Accepted at IEEE/ACM CCGrid 2026
## Pipeline
The implementation follows these steps:

1. Extract intermediate features from VGG19 (layer 20)
2. Generate adversarial samples (external method)
3. Construct original and mixed datasets
4. Add noise with different levels
5. Train adVAE on noisy original data
6. Extract detection features using the trained adVAE
7. Generate CSV files for original and mixed data
8. Split mixed data into validation and test sets
9. Train and tune OC-SVM
10. Evaluate on test data

## Repository Structure
- `src/` : core implementation (data processing, noise, adVAE, OC-SVM, evaluation)
- `scripts/` : runnable scripts for each stage of the pipeline
- `configs/` : configuration files (model, layer, noise settings)
- `models/` : saved models (VGG19,adVAE, OC-SVM)
- `data/` : intermediate and processed data (features, noisy data, CSV files)
- `results/` : outputs, logs, and evaluation results

## Status
This repository is under development

## Citation
Paper: https://arxiv.org/abs/2603.17914

If you use this work, please cite:

@article{yousefi2026noiseaware,
  title={Noise-Aware Misclassification Attack Detection in Collaborative DNN Inference},
  author={Yousefi, Shima and Debroy, Saptarshi},
  journal={arXiv preprint arXiv:2603.17914},
  year={2026},
  doi={10.48550/arXiv.2603.17914}
}

## Contact
For questions, please open an issue in this repository.
