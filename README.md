# Fake News Classification: Comparative Analysis with Bidirectional LSTM and ALBERT

This project focuses on the comparative analysis of two advanced approaches to tackling the fake news classification problem: a **Bidirectional LSTM (Long Short-Term Memory) model** trained from scratch and **fine-tuning the ALBERT (A Lite BERT) model**. Both models are quantized to enhance efficiency. The goal is to evaluate and compare the performance of these two methods in terms of accuracy, efficiency, and scalability.

## Project Overview

Fake news classification is a challenging task in natural language processing. This project aims to explore two distinct methods for fake news detection:
1. **Bidirectional LSTM**: A sequential deep learning model capable of learning long-term dependencies.
2. **ALBERT Fine-Tuning**: A transformer-based model known for its light-weight and efficient architecture.

The evaluation of these models includes comparisons based on:
- **Accuracy**: How well the models classify fake vs. real news.
- **Efficiency**: Memory and computational efficiency, especially with quantization applied.
- **Scalability**: Performance when deployed at scale.

## Dataset

The dataset used in this project consists of news articles labeled as real or fake. Detailed preprocessing and cleaning steps are included in the notebooks. Please ensure you have the dataset downloaded to use the notebooks effectively.

https://www.kaggle.com/datasets/evilspirit05/english-fake-news-dataset


## Notebooks

1. **01-Dataset Analysis and Preprocessing.ipynb**: 
   - Preprocesses and cleans the dataset, including tokenization, stopword removal, and lemmatization.
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/starryendymion/ML-GP1-Monsoon-2024-SOEJNU/blob/main/notebooks/1_Data_Preprocessing.ipynb)

2. **02-Training-CASE1.ipynb**:
   - Builds and trains a Bidirectional LSTM model from scratch, with quantization applied.
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/starryendymion/ML-GP1-Monsoon-2024-SOEJNU/blob/main/notebooks/2_Bidirectional_LSTM_Training.ipynb)

3. **03-Training(CASE2).ipynb**:
   - Fine-tunes the ALBERT model on the fake news dataset, with quantization applied.
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/starryendymion/ML-GP1-Monsoon-2024-SOEJNU/blob/main/notebooks/3_ALBERT_Fine_Tuning.ipynb)

4. **04-Inference and Evaluation.ipynb**:
   - Compares the models based on accuracy, efficiency, and scalability. Generates performance metrics and visualizations.
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/starryendymion/ML-GP1-Monsoon-2024-SOEJNU/blob/main/notebooks/4_Evaluation_Comparison.ipynb)


