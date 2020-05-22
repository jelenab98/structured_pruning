import torch
import torch.nn as nn
import torch_pruning as pruning
import norms

def pruning_plan_vgg19_cifar(model, DG, pr_percentage):
    pruning_plans = []
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for layer_seq in layer.children():
                if isinstance(layer_seq, nn.Conv2d):
                    params = list(layer_seq.parameters())[0]
                    probs = norms.filters_lp_norm(params)
                    if len(probs) <= 1:
                        continue

                    probs_sorted_idx = torch.argsort(probs)
                    n_pruned = int(len(probs) * pr_percentage)
                    if n_pruned == 0:
                        n_pruned = 1
                    myidxs = probs_sorted_idx[:n_pruned].tolist()
                    pruning_plans.append(DG.get_pruning_plan( layer_seq, pruning.prune_conv, idxs=myidxs ))

    return pruning_plans

