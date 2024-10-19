# ScaleMatch
This repository contains the official implementation of the ScaleMatch.
## Getting Started

### Environment
Create a new environment and install the requirements:
```shell
conda create -n scalematch python=3.10
conda activate scalematch
pip install -r requirements.txt
```
### Dataset:
Please modify the dataset path in configuration files.*The groundtruth mask ids have already been pre-processed. You may use them directly.*

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

### Pascal VOC 2012 original

Labeled images are sampled from the **original high-quality** training set. Results are obtained by DeepLabv3+ based on ResNet-101 with training size 513.

|        Method        | 1/16 (92) | 1/8 (183) |   1/4 (366)    | 1/2 (732) | Full (1464) |
|:--------------------:|:---------:|:---------:|:--------------:|:---------:|:-----------:|
|     SupBaseline      |   48.3    |   56.2    |      66.6      |   71.3    |    75.4     |
|       UniMatch       |   75.2    |   77.2    |      78.8      |   79.9    |    81.2     |
|       Allspark       |   76.0    |   78.4    |      79.7      |   80.8    |    82.1     |
|       RankMatch      |   75.5    |   77.6    |      79.8      |   80.7    |    82.2     |
| **ScaleMarch (Ours)** | **76.7**  | **78.6**  |    **80.5**    | **82.0**  |  **83.0**   |


### Pascal VOC 2012 augmented

Labeled images are sampled from the **original high-quality** training set. Results are obtained by DeepLabv3+ based on ResNet-101 with training size 513, ♢ means using the same split as U2PL.

|        Method        | 1/16 (662) | 1/8 (1323) |   1/4 (366)    |
|:--------------------:|:---------:|:---------:|:--------------:|
|     SupBaseline      |   67.2    |   70.6    |      73.8      |
|       UniMatch       |   78.1    |   78.4    |      79.2      |
|       RankMatch      |   78.9    |   79.2    |      80.0      |
|       CorrMatch      |   78.4    |   79.3    |      79.6      |
| **ScaleMarch (Ours)** | **78.6**  | **79.5**  |    **80.2**    |
|:--------------------:|:---------:|:---------:|:--------------:|
|     SupBaseline♢     |   70.6    |   75.0    |      76.5      |
|         U2PL♢        |   77.2    |   79.0    |      79.3      |
|       UniMatch       |   80.9    |   81.9    |      80.4      |
|       Allspark       |   80.6    |   82.0    |      80.9      |
|       CorrMatch      |   81.3    |   81.9    |      80.9      |
| **ScaleMarch (Ours)** | **81.5**  | **82.7**  |    **81.1**    |


### Cityscapes

Results are obtained by DeepLabv3+ based on ResNet-101.

|        Method        | 1/16 (186) | 1/8 (372) | 1/4   (744) | 1/2 (1488) |
|:--------------------:|:----------:|:---------:|:-----------:|:----------:|
|       SupOnly        |    65.7    |   72.5    |    74.4     |    77.8    |
|       UniMatch       |    76.6    |   77.9    |   79.2      |    79.5    |
|       CorrMatch      |    77.3    |   78.5    |   79.4      |    80.4    |
| **ScaleMarch (Ours)** |  **77.8**  | **79.4**  |  **80.2**  |  **80.9**  |


