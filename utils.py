from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import gdown

import re
import nltk
import json
import pandas as pd



def process_text(text,tokenizer):

  lemmatizer = WordNetLemmatizer()
  stop_words = set(stopwords.words('english'))
  text = text.lower()
  text = re.sub(r'\W', ' ', text)
  tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
  query=[' '.join(tokens)]

  max_length = 300
  sequences = tokenizer.texts_to_sequences(query)
  padded_features = pad_sequences(sequences, maxlen=max_length, padding='post')

  return padded_features

def infer(data,bi_tokenizer,bi_model,al_model):

    result=[]
    data_bi=process_text(data,bi_tokenizer)

    result.append(round(bi_model.predict(data_bi)[0][0]))
    result.append(round(al_model.predict([data])[0][0]))

    return result

def bulk_inference(dataset, bi_tokenizer, bi_model, al_model):
    model1_predictions = []
    model2_predictions = []

    for feature in dataset:
        # Get predictions for each model
        predictions = infer(feature, bi_tokenizer, bi_model, al_model)

        # Append predictions to respective lists
        model1_predictions.append(predictions[0])  # Prediction from model 1
        model2_predictions.append(predictions[1])  # Prediction from model 2

    return model1_predictions, model2_predictions


def evaluate(features, labels, bi_tokenizer, bi_model, al_model):
    # Get predictions for both models
    model1_preds, model2_preds = bulk_inference(features, bi_tokenizer, bi_model, al_model)

    # Calculate confusion matrix for both models
    cm_model1 = confusion_matrix(labels, model1_preds)
    cm_model2 = confusion_matrix(labels, model2_preds)

    # Calculate ROC AUC for both models
    roc_auc_model1 = roc_auc_score(labels, model1_preds)
    roc_auc_model2 = roc_auc_score(labels, model2_preds)

    # Plot confusion matrix for Model 1
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm_model1, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix for Model 1")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Plot confusion matrix for Model 2
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_model2, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title("Confusion Matrix for Model 2")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

    # Calculate ROC curve for both models
    fpr1, tpr1, _ = roc_curve(labels, model1_preds)
    fpr2, tpr2, _ = roc_curve(labels, model2_preds)

    # Plot ROC curves
    plt.figure()
    plt.plot(fpr1, tpr1, label=f"Model 1 (AUC = {roc_auc_model1:.2f})")
    plt.plot(fpr2, tpr2, label=f"Model 2 (AUC = {roc_auc_model2:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.show()

    # Print other metrics for Model 1
    print("Model 1 Evaluation Metrics:")
    print("Accuracy:", accuracy_score(labels, model1_preds))
    print("Precision:", precision_score(labels, model1_preds))
    print("Recall:", recall_score(labels, model1_preds))
    print("F1 Score:", f1_score(labels, model1_preds))
    print("ROC AUC Score:", roc_auc_model1)

    # Print other metrics for Model 2
    print("\nModel 2 Evaluation Metrics:")
    print("Accuracy:", accuracy_score(labels, model2_preds))
    print("Precision:", precision_score(labels, model2_preds))
    print("Recall:", recall_score(labels, model2_preds))
    print("F1 Score:", f1_score(labels, model2_preds))
    print("ROC AUC Score:", roc_auc_model2)

    # Plot error curves (Accuracy over iterations)
    plt.figure()
    acc_model1 = accuracy_score(labels, model1_preds)
    acc_model2 = accuracy_score(labels, model2_preds)
    plt.plot([1, 2], [acc_model1, acc_model2], marker='o', label="Accuracy")
    plt.title("Error Curves")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.xticks([1, 2], ["Model 1", "Model 2"])
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
