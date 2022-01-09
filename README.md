# Multimodal emotion analysis on social media (master thesis project).

## Table of Contents
* [Dataset](#Dataset)
* [Install](#Install)

## Dataset
The dataset is to be released after the paper has been published, and can be found in [data](data/). Until the dataset is published, we created mock files to reflect the structure of it. The multilabel dataset is stored in .csv format and split into training and test datasets (90/10). Each .csv file has *image*, *text* columns, as well as columns for labels (each label is prefixed with either *emotion*, *relation*, or *stimulus*). The images are stored in [images](data/images).
The dataset consists of 1061 multimodal social media posts collected from Reddit and annotated for emotions, text-image relations, and emotion stimuli, using Amazon Mechanical Turk. The exact annotation procedure is described in the paper.

## Install
This project uses 
