# Multimodal emotion analysis on social media (master thesis project).

## Table of Contents
* [Description](#Description)
* [Dataset](#Dataset)
* [Install](#Install)

## Description
This project aims to analyze emotions in social media posts. We collected multimodal posts (text + image) from Reddit and annotated them for emotions, text-image relations (e.g. complementary), and emotion stimuli (e.g. animal). We then created computational models using RoBERTa from Hugging Face for the text modality, and ResNet50 from torchvision for the image modality. In total, we implemented 5 models in the multitask setting (all three labels are predicted at the same time):
* text-based model (pretrained RoBERTa model)
* image-based model (pretrained ResNet50 model with frozen layers)
* early fusion model (tokenized text and image transformed to tensor are joined in the early stage and passed through several linear layers)
* late fusion model (outputs of the trained text-based and image-based models are joined and passed through the final layers, without updating the weights of the simgle-modality models)
* model-based model (final hidden layers of the text-based and image-based models are joined and passed through the a small neural network on top, updating the weights of both text-based and image-based models)

## Dataset
The dataset is to be released after the paper has been published, and can be found in [data](data/). Until the dataset is published, we created mock files to reflect the structure of it. The multilabel dataset is stored in .csv format and split into training and test datasets (90/10). Each .csv file has *image*, *text* columns, as well as columns for labels (each label is prefixed with either *emotion*, *relation*, or *stimulus*). The images are stored in [images](data/images).
The dataset consists of 1061 multimodal social media posts collected from Reddit and annotated for emotions, text-image relations, and emotion stimuli, using Amazon Mechanical Turk. The exact annotation procedure is described in the paper.

## Install
This project uses [torch](https://pytorch.org/get-started/locally/), [transformers](https://huggingface.co/docs/transformers/index), [PIL](https://pillow.readthedocs.io/en/stable/index.html), [sklearn](https://pypi.org/project/scikit-learn/), [pandas](https://pypi.org/project/pandas/), [numpy](https://numpy.org), [html](https://pypi.org/project/html/), [emoji](https://pypi.org/project/emoji/), and [argparse](https://pypi.org/project/argparse/). If you do not have them installed, please use the following commands:

```
$ pip install torch torchvision
$ pip install transformers
$ python -m pip install --upgrade Pillow
$ pip install -U scikit-learn
$ pip install pandas
$ pip install numpy
$ pip install html
$ pip install emoji
$ pip install argparse
```
