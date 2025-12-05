"""
utils.py

Helper functions shared by the training scripts, such as
metric computation and dataset loading. In our project,
the full implementations live in the Colab notebook, and
this file serves as a lightweight reference.
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

def compute_metrics(eval_pred):
    """
    Example metric function consistent with the Colab notebook:
    - micro F1
    - macro F1
    - AUC
    """
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))   # sigmoid
    preds = (probs > 0.5).astype(int)

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        auc = 0.0

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "auc": auc,
    }

def load_datasets(tokenizer):
    """
    Placeholder function.

    In practice, dataset loading, splitting (70/15/15) and
    tokenization are all done inside the Colab notebook.
    Here we only keep a stub so the project structure is clear.
    """
    raise NotImplementedError(
        "Datasets are prepared in the Colab notebook "
        "(see notebook/another_copy_of_final_project.ipynb)."
    )
