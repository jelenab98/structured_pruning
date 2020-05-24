import torch
import torch.nn as nn
import torch_pruning as pruning
import norms
import json
from utils import merge_prune_indexes

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



class PruningPlanLottery:
    def __init__(self, model, pr):
        self.pruning_plan = {}
        for module_ in model.named_modules():
            if isinstance(module_[1], nn.Conv2d):
                self.pruning_plan[module_[0]] = (len(list(module_[1].parameters())[0]), [])
        
        self.pr = pr
    

    def save_pp(self, save_path):
        with open(save_path, 'w') as fp:
            json.dump(self.pruning_plan, fp, sort_keys=True, indent=4)

    def load_pp(self, load_path):
        with open(load_path, 'r') as fp: 
            self.pruning_plan = json.load(fp)



    def _append_pruning_plan(self, layer_name, pruning_candidates):
            self.pruning_plan[layer_name] = (self.pruning_plan[layer_name][0], merge_prune_indexes(self.pruning_plan[layer_name][0], self.pruning_plan[layer_name][1], pruning_candidates))
       

    def get_next_pruning_plan(self, model):
        pruning_plans = []

        for module_ in model.named_modules():
            if isinstance(module_[1], nn.Conv2d):
                params = list(module_[1].parameters())[0]
                probs = norms.filters_lp_norm(params)
                if len(probs) <= 1:
                    continue

                probs_sorted_idx = torch.argsort(probs)
                n_pruned = int(len(probs) * self.pr)
                if n_pruned == 0:
                    n_pruned = 1
                myidxs = probs_sorted_idx[:n_pruned].tolist()
                self._append_pruning_plan(module_[0], myidxs) 


    def prune_by_pruning_plan(self, model, DG, pr_percentage):
        pruning_plans = []
        

        for module_ in model.named_modules():
            if isinstance(module_[1], nn.Conv2d):
                pruning_plans.append(DG.get_pruning_plan( module_[1], pruning.prune_conv, idxs=self.pruning_plan[module_[0]][1]))

        return pruning_plans


