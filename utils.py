import torch
import torch.nn as nn
import torch_pruning as pruning
from vgg_cifar import vgg19_cifar
import numpy as np
import logging
import json


msglogger = logging.getLogger()

def num_of_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def normalize_module_name(layer_name):
    """Normalize a module's name.

    PyTorch let's you parallelize the computation of a model, by wrapping a model with a
    DataParallel module.  Unfortunately, this changs the fully-qualified name of a module,
    even though the actual functionality of the module doesn't change.
    Many time, when we search for modules by name, we are indifferent to the DataParallel
    module and want to use the same module name whether the module is parallel or not.
    We call this module name normalization, and this is implemented here.
    """
    modules = layer_name.split('.')
    try:
        idx = modules.index('module')
    except ValueError:
        return layer_name
    del modules[idx]
    return '.'.join(modules)


def normalize_state_dict(state_dict):
     new_keys = []
     for _key in state_dict.keys():
         new_keys.append((_key, normalize_module_name(_key)))
     for new_old_k in new_keys:
         state_dict[new_old_k[1]] = state_dict.pop(new_old_k[0])


def denormalize_module_name(parallel_model, normalized_name):
    """Convert back from the normalized form of the layer name, to PyTorch's name
    which contains "artifacts" if DataParallel is used.
    """
    fully_qualified_name = [mod_name for mod_name, _ in parallel_model.named_modules() if
                            normalize_module_name(mod_name) == normalized_name]
    if len(fully_qualified_name) > 0:
        return fully_qualified_name[-1]
    else:
        return normalized_name   # Did not find a module with the name <normalized_name>


def get_pretrained_vgg19(model_path):
    model = vgg19_cifar()
    checkpoint_data = torch.load(model_path)
    net_kyw = 'state_dict'
    if checkpoint_data['state_dict'] == None:
        net_kyw = 'net'
    normalize_state_dict(checkpoint_data[net_kyw])
    model.load_state_dict(checkpoint_data[net_kyw])
    return model


def hardprune_f(model, calc_prun_cand, pr_percentage=0.1):
    DG = pruning.DependencyGraph( model, fake_input=torch.randn(1,3,32,32) )
    # get a pruning plan according to the dependency graph. idxs is the indices of pruned filters.
    
    pruning_plan_list = calc_prun_cand(model, DG, pr_percentage) 
    # execute this plan (prune the model)
    for i, pp in enumerate(pruning_plan_list):
        # print("=========== ", i, " ============ ")
        # print(pp)
        pp.exec()
    
        
def merge_prune_indexes(len_, prune_1, prune_2):
    resulting_list = list(range(0,len_))
    
    for p in prune_1:
        resulting_list.remove(p)
    
    for p in prune_2:
        prune_1.append(resulting_list[p])
    
    prune_1.sort()
    return prune_1




