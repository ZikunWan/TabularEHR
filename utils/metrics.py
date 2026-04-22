import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def calc_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def calc_auroc(y_true, y_prob, *, multi_class=None, average=None):
    try:
        kwargs = {}
        if multi_class is not None:
            kwargs["multi_class"] = multi_class
        if average is not None:
            kwargs["average"] = average
        return roc_auc_score(y_true, y_prob, **kwargs)
    except ValueError:
        return 0.5


def calc_f1(y_true, y_pred, *, average="binary"):
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def calc_recall(y_true, y_pred, *, average="binary"):
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_classification_metrics(eval_pred):
    """
    Unified metric function for HuggingFace Trainer.

    Returns:
        {
            "auroc": float,
            "accuracy": float,
            "f1": float,
            "recall": float,
        }
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits)
    labels = np.asarray(labels)
    y_true = labels.reshape(-1).astype(int)

    # Binary classification: logits shape [N] or [N, 1]
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[-1] == 1):
        probs = _sigmoid(logits.reshape(-1))
        preds = (probs > 0.5).astype(int)

        auroc = calc_auroc(y_true, probs)
        acc = calc_accuracy(y_true, preds)
        f1 = calc_f1(y_true, preds, average="binary")
        recall = calc_recall(y_true, preds, average="binary")
        return {"auroc": auroc, "accuracy": acc, "f1": f1, "recall": recall}

    # Binary classification: two-logit format [N, 2]
    if logits.ndim == 2 and logits.shape[-1] == 2 and labels.ndim == 1:
        probs_2d = _softmax(logits, axis=-1)
        probs = probs_2d[:, 1]
        preds = np.argmax(probs_2d, axis=-1).astype(int)

        # In case labels are not exactly {0,1}, remap to {0,1} by sorted order.
        unique_labels = np.unique(y_true)
        if unique_labels.size == 2 and not np.array_equal(unique_labels, np.array([0, 1])):
            remap = {int(unique_labels[0]): 0, int(unique_labels[1]): 1}
            y_true = np.array([remap[int(v)] for v in y_true], dtype=int)

        auroc = calc_auroc(y_true, probs)
        acc = calc_accuracy(y_true, preds)
        f1 = calc_f1(y_true, preds, average="binary")
        recall = calc_recall(y_true, preds, average="binary")
        return {"auroc": auroc, "accuracy": acc, "f1": f1, "recall": recall}

    # Multi-label classification: logits and labels both [N, C]
    if labels.ndim == logits.ndim and labels.shape == logits.shape and logits.ndim == 2:
        probs = _sigmoid(logits)
        preds = (probs > 0.5).astype(int)
        y_true = labels.astype(int)

        auroc = calc_auroc(y_true, probs, average="micro")
        acc = calc_accuracy(y_true, preds)
        f1 = calc_f1(y_true, preds, average="micro")
        recall = calc_recall(y_true, preds, average="micro")
        return {"auroc": auroc, "accuracy": acc, "f1": f1, "recall": recall}

    # Multi-class classification: logits [N, C], labels [N]
    probs = _softmax(logits, axis=-1)
    preds = np.argmax(probs, axis=-1)

    auroc = calc_auroc(y_true, probs, multi_class="ovr")
    acc = calc_accuracy(y_true, preds)
    f1 = calc_f1(y_true, preds, average="macro")
    recall = calc_recall(y_true, preds, average="macro")
    return {"auroc": auroc, "accuracy": acc, "f1": f1, "recall": recall}


def compute_metrics(eval_pred):
    return compute_classification_metrics(eval_pred)
