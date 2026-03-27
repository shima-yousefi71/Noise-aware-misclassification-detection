# Noise-Aware Misclassification Attack Detection in Collaborative DNN Inference

## Overview
This repository provides the implementation of a noise-aware framework for detecting adversarial misclassification attacks in collaborative DNN inference.

Accepted at IEEE/ACM CCGrid 2026.

The method operates on intermediate features extracted from a VGG19 model (layer 20) trained on CIFAR-100. It considers both adversarial manipulation and noise during detection. The current public version focuses on the VGG19 layer 20 setting.

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
9. Train and tune One-Class SVM (OC-SVM)
10. Predict on the test set
11. Evaluate detection results

## Repository Structure
- `src/` : core implementation
- `scripts/` : runnable scripts for each stage of the pipeline
- `models/` : saved models and trained artifacts
- `data/` : intermediate and processed feature data
- `results/` : generated CSV files, predictions, plots, and evaluation outputs

### Main Scripts
- `scripts/train_vgg19.py`
- `scripts/collect_features.py`
- `scripts/add_noise.py`
- `scripts/train_advae.py`
- `scripts/extract_detection_features.py`
- `scripts/split_dataset.py`
- `scripts/train_ocsvm.py`
- `scripts/predict_ocsvm.py`
- `scripts/evaluate_ocsvm.py`
## Usage
The main workflow is:

1. Train or load the feature extractor
2. Collect intermediate features
3. Add noise
4. Train adVAE
5. Extract detection features
6. Split validation and test sets
7. Train OC-SVM
8. Predict on test data
9. Evaluate results
10. 
## Status
This repository is under development. The current version provides the main pipeline for the VGG19 layer 20 setting.

## Citation
Paper: https://arxiv.org/abs/2603.17914

If you use this work, please cite:

```bibtex
@article{yousefi2026noiseaware,
  title={Noise-Aware Misclassification Attack Detection in Collaborative DNN Inference},
  author={Yousefi, Shima and Debroy, Saptarshi},
  journal={arXiv preprint arXiv:2603.17914},
  year={2026},
  doi={10.48550/arXiv.2603.17914}
}
```

## Contact
For questions, please open an issue in this repository.
