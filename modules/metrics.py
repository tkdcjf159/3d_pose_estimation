"""
"""
import torch
import pdb
def get_metric_fn(y_pred, y_answer):
    """ Metric 함수 반환하는 함수

    Returns:
        metric_fn (Callable)
    """
    #y_pred = torch.cat(y_pred)
    #y_answer = torch.cat(y_answer) 
    #mpjpe = torch.pow(y_pred - y_answer, 2).sum(dim=2).mean(dim=1).mean().item()
    #mpjpe = torch.pow(y_pred - y_answer, 2).mean(dim=2).mean(dim=1).mean().item()
    mpjpe = torch.pow(y_pred - y_answer, 2).mean(dim=2).mean(dim=1).mean().item()
    return mpjpe

