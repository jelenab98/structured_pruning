import torch
import torch.nn as nn
import torchvision.models as models
import torch_pruning as pruning
import norms as norms
from vgg_cifar import vgg19_cifar
from cifar_train import CifarTrain
import time
import sys
import os
import numpy as np
import pdb
import logging
from thop import profile


log_time = time.strftime("%Y%m%d-%H%M%S")
if not os.path.isdir('./logs'):
    os.mkdir('./logs')

if not os.path.isdir('./logs/' + log_time):
    os.mkdir('./logs/' + log_time)

logging.basicConfig(filename="logs/" + log_time + "/out.log", level=logging.NOTSET)
stdout_handler = logging.StreamHandler(sys.stdout)
msglogger = logging.getLogger()
msglogger.addHandler(stdout_handler)

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


def get_pretrained_vgg19():
    model = vgg19_cifar()
    checkpoint_data = torch.load('/home/smocilac/distiller/examples/classifier_compression/logs/2020.05.16-160108/best.pth.tar')
    normalize_state_dict(checkpoint_data['state_dict'])
    model.load_state_dict(checkpoint_data['state_dict'])
    return model


def hardprune_f(model, calc_prun_cand, pr_percentage=0.1):
    # build layer dependency for resnet18
    DG = pruning.DependencyGraph( model, fake_input=torch.randn(1,3,32,32) )
    # get a pruning plan according to the dependency graph. idxs is the indices of pruned filters.
    
    pruning_plan_list = calc_prun_cand(model, DG, pr_percentage) 
    # execute this plan (prune the model)
    for i, pp in enumerate(pruning_plan_list):
        # print("=========== ", i, " ============ ")
        # print(pp)
        pp.exec()
    

def calculate_pruning_candidates(model, DG, pr_percentage):
    pruning_plans = []
    for layer in model.children():

        if isinstance(layer, nn.Conv2d):
            params = list(layer.parameters())[0]
            probs = norms.filters_lp_norm(params)
            if len(probs) <= 1:
                continue
            
            probs_sorted_idx = torch.argsort(probs)
            n_pruned = int(len(probs) * pr_percentage)
            if n_pruned == 0:
                n_pruned = 1
            myidxs = probs_sorted_idx[:n_pruned].tolist()
            pruning_plans.append(DG.get_pruning_plan( layer, pruning.prune_conv, idxs=myidxs ))

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





def get_eval_time(model, device): 
    random_inputs = []
    for i in range(0,101):
        random_inputs.append(torch.randn(1, 3, 32, 32).to(device))
    model(random_inputs[0])

    t1b = time.perf_counter()
    for j in range(0,10):
        for i in range(1,101):
            model(random_inputs[i])
            if device != 'cpu':
                torch.cuda.synchronize()
    t2 = time.perf_counter()
    
    return 1000.0 * (t2 - t1b) / 1000.0


def main():
    # load pretrained model
    model = get_pretrained_vgg19()

    prune_at_idx = [0,1,2,3,4,5,6,7,8,9]
    
    msglogger.info("Number of parameters: {}".format(num_of_trainable_params(model)))
    
    for i in range(0,11):
        input = torch.randn(1, 3, 32, 32)
        macs, params = profile(model, inputs=(input, ))
        pdb.set_trace()
        
        hardprune_f(model, calculate_pruning_candidates,0.1)
        
        msglogger.info("MACs {}  ;  Params {}".format(macs, params))
        msglogger.info("Number of parameters: {}".format(num_of_trainable_params(model)))


if __name__ == '__main__':
    main()



