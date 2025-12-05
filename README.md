# -iat360-Final-Project-group2-
Overview

This project fine-tunes a DistilBERT model to detect toxic language in online comments using the Kaggle Jigsaw Toxic Comment Classification dataset. The model predicts six toxicity categories at the same time:

- toxic

- severe_toxic

- obscene

- threat

- insult

- identity_hate

This is a multi-label classification task. The goal is to explore how transformer models can help identify harmful language and improve online safety.

Project Setup
Requirements

- Python 3.10

- GPU

Libraries:

pip install transformers datasets torch scikit-learn pandas numpy matplotlib

Dataset

Download from Kaggle:
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

Place train.csv in a directory and update the path in the notebook if needed.



How to Run the Project

1. Load and Explore Data

The notebook loads:

- First 5 rows

- Label distribution

- Basic text checks

2. Train/Validation/Test Split

- 70% training

- 15% validation

- 15% test

3. Tokenization

- We use DistilBertTokenizerFast with:

- max length = 128

- truncation + padding

4. Model Training

- We fine-tune DistilBertForSequenceClassification using:

- 2 epochs

- batch size = 16

- mixed precision (FP16)

- binary cross-entropy (HuggingFace default for multi-label tasks)

5. Evaluation

- We report:

- Micro F1

- Macro F1

- ROC-AUC

- Per-label precision/recall/F1

- Example predictions

- Threshold tuning results

Results Summary

Validation (Epoch 2):
Metric	Score
Micro F1	~0.76
Macro F1	~0.66
AUC	~0.988

Test Set:
Metric	Score
Micro F1	~0.79
Macro F1	~0.65
AUC	~0.99

Threshold Tuning:

The best global threshold was 0.7, improving macro F1 slightly.


Key Findings

- The model detects toxic, obscene, and insult reliably.

- It struggles with rare labels like threat, identity_hate, and severe_toxic because they appear much less often in the dataset.

- AUC is very high → the model separates toxic vs. non-toxic comments well.

- Threshold tuning helps improve macro F1.

Bias, Ethics, and Limitations

- Identity-related words (e.g., “gay”, “Muslim”) may cause false positives.

- Dataset imbalance leads to weaker performance on rare labels.

- Models like this should not be used alone for moderation — they must include human review.
