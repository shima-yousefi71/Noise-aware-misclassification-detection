# Noise-Aware Misclassification Attack Detection in Collaborative DNN Inference

## Overview
This repository provides the implementation of a noise-aware framework for detecting adversarial misclassification attacks in collaborative DNN inference.
Accepted at IEEE/ACM CCGrid 2026

The method operates on intermediate features extracted from a VGG19 model (layer 20) trained on CIFAR-100. It considers both adversarial manipulation and noise during detection.

## Pipeline
The implementation follows these steps:

1. Retrain a pretrained VGG19 model on CIFAR-100
2. Collect intermediate features from VGG19 layer 20
3. Generate adversarial intermediate samples
4. Build original and mixed datasets
5. Add noise with different levels
6. Train adVAE on noisy original data
7. Extract detection features and generate CSV files
8. Split mixed data into validation and test sets
9. Train and tune OC-SVM
10. Evaluate on the test set

## Repository Structure
- `src/` : core implementation
- `scripts/` : runnable files for each stage
- `models/` : saved models (VGG19, adVAE, OC-SVM)
- `data/` : intermediate and processed data
- `results/` : outputs and evaluation results

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
