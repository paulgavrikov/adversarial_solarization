import torch


def ce_loss(logits, labels):
    """
    Cross-entropy loss
    """
    probs = torch.softmax(logits, dim=1)
    loss = probs.gather(1, labels.unsqueeze(1)).squeeze(1).log().neg().mean()
    return loss


def cw_loss(logits, labels):
    """
    Carlini-Wagner loss
    """
    probs = torch.softmax(logits, dim=1)

    onehot_labels = torch.eye(len(probs[0]))[labels].to(probs.device)

    real = (onehot_labels * probs).sum(1)
    other = ((1.0 - onehot_labels) * probs - onehot_labels * 10000.0).max(1)[0]

    loss = torch.clamp(other - real + 50.0, min=0.0).mean()
    return loss