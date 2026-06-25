# PLCSeg: sar-guided progressive learning for semantic segmentation of cloudy remote sensing images
## Introduction

Cloud contamination is a common problem in optical remote sensing images, which can seriously affect the performance of semantic segmentation models. To address this issue, this project proposes **PLCSeg**, a SAR-guided progressive learning method for semantic segmentation of cloudy remote sensing images.

The proposed method uses SAR images as auxiliary guidance to improve the feature representation of cloudy optical images and progressively enhance the segmentation performance under cloud-covered conditions.

## Features

* Semantic segmentation of cloudy remote sensing images
* SAR-guided feature learning
* Progressive learning strategy
* Cloud-robust segmentation framework
* Python-based implementation

## Project Structure

```text
PLCSeg/
├── model/
├── CloudRemoval.py
├── dataloader.py
├── model_base.py
├── trainseedMaskDice.py
├── utils.py
├── README.md
└── LICENSE
```

## Requirements

The code is implemented in Python. Please install the required dependencies before running the project.

Example environment:

```text
Python >= 3.8
PyTorch >= 1.8
NumPy
OpenCV
tqdm
scikit-learn
```

You can install the required packages using:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided yet, please install the dependencies manually according to your running environment.

## Dataset

Please prepare the cloudy remote sensing dataset and SAR images before training.

The dataset should include:

```text
dataset/
├── optical_images/
├── sar_images/
├── labels/
└── train_val_test_split/
```

You can modify the dataset path in:

```text
dataloader.py
```

## Usage

### Training

Run the following command to train the model:

```bash
python trainseedMaskDice.py
```

### Testing

The testing code will be released after the project is fully organized.

## Results

Experimental results will be added after the paper is accepted.

## Citation

If you find this project useful for your research, please cite our paper:

```bibtex
@article{PLCSeg,
  title={PLCSeg: SAR-guided Progressive Learning for Semantic Segmentation of Cloudy Remote Sensing Images},
  author={},
  journal={},
  year={}
}
```

## Code Availability

The code will be published after the paper is accepted.

## License

This project is released under the MIT License.

## Contact

If you have any questions, please feel free to contact the author.

```text
Author: wangzhuo1998
GitHub: https://github.com/wangzhuo1998
```
