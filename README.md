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
