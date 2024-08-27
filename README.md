# CorrMatch

This repository contains the official implementation of the ScaleMatch.

## Getting Started


### Environment
Create a new environment and install the requirements:
```shell
pip install -r requirements.txt
```

### Dataset:

Your dataset path may look like:
```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
```

## Usage

### Training ScaleMatch

```bash
sh scripts/train.sh <num_gpu> <port>
```
To run on different labeled data partitions or different datasets, please modify:

``config``, ``labeled_id_path``, ``unlabeled_id_path``, and ``save_path`` in [train.sh]

### Evaluation
```bash
sh scripts/inference.sh <num_gpu> <port>
```
To evaluate your checkpoint, please modify ``checkpoint_path`` in [inference.sh]

## Results

### Pascal VOC 2012

Labeled images are sampled from the **original high-quality** training set. Results are obtained by DeepLabv3+ based on ResNet-101 with training size 321(513).

|        Method        | 1/16 (92) | 1/8 (183) |   1/4 (366)    | 1/2 (732) | Full (1464) |
|:--------------------:|:---------:|:---------:|:--------------:|:---------:|:-----------:|
|       SupOnly        |   45.1    |   55.3    |      64.8      |   69.7    |    73.5     |
|         ST++         |   65.2    |   71.0    |      74.6      |   77.3    |    79.1     |
|        PS-MT         |   65.8    |   69.6    |      76.6      |   78.4    |    80.0     |
|       UniMatch       |   75.2    |   77.2    |      78.8      |   79.9    |    81.2     |
| **ScaleMarch (Ours)** | **76.7**  | **78.6**  |    **80.5**    | **82.1**  |  **83.0**   |


### Cityscapes

Results are obtained by DeepLabv3+ based on ResNet-101.

|        Method        | 1/16 (186) | 1/8 (372) | 1/4   (744) | 1/2 (1488) |
|:--------------------:|:----------:|:---------:|:-----------:|:----------:|
|       SupOnly        |    65.7    |   72.5    |    74.4     |    77.8    |
|       UniMatch       |    76.6    |   77.9    |   79.2     |    79.5    |
| **ScaleMarch (Ours)** |  **77.8**  | **79.4**  |  **80.2**   |  **80.9**  |


