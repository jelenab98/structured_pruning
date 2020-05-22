import torch
from cifar_train import CifarTrain
import time
import sys
import os
import pdb
import logging
from datetime import datetime 
from utils import num_of_trainable_params, get_pretrained_vgg19, hardprune_f

from pruning_plans import pruning_plan_vgg19_cifar

log_time = time.strftime("%Y%m%d-%H%M%S")
if not os.path.isdir('./logs'):
    os.mkdir('./logs')

if not os.path.isdir('./logs/' + log_time):
    os.mkdir('./logs/' + log_time)

logging.basicConfig(filename="logs/" + log_time + "/out.log", level=logging.NOTSET)
stdout_handler = logging.StreamHandler(sys.stdout)
msglogger = logging.getLogger()
msglogger.addHandler(stdout_handler)


def main():
    # init train class
    batch_sz = 2
    prune_p_params = 0.05
    model_path = '/home/smocilac/hard_prune/logs/20200520-125605/checkpoint/ckpt.pth'
    cifar_train = CifarTrain(log_time, batch_sz)

    # select train and test device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load pretrained model
    model = get_pretrained_vgg19(model_path).to(device)
    
    model = model.cpu()
    hardprune_f(model, pruning_plan_vgg19_cifar, prune_p_params)
    model = model.to(device)
    msglogger.info("Number of parameters: {}".format(num_of_trainable_params(model)))
    

    cifar_train.initialize(model, lr=2e-5)
    
    not_improved_for = 0
    current_best = 0.0
    cntr = 9

    t_begin = datetime.now()

    while True:
        t0 = datetime.now()
        cifar_train.train(cntr, 50000 // batch_sz)
        t1 = datetime.now()
        acc = cifar_train.test(cntr, 10000 // batch_sz)
        t2 = datetime.now()
        cntr += 1

        msglogger.info("[hh:mm:ss.ms] Train time {}  ;  Test time {}  ;  Time elapsed {}".format(t1-t0, t2-t1, t2-t_begin))
        
        if acc < current_best:
            not_improved_for = not_improved_for + 1
        else:
            not_improved_for = 0
            current_best = acc

        if not_improved_for >= 3:
            not_improved_for = 0
            current_best = 0.0
            msglogger.info("Pruning p={}% of parameters".format(prune_p_params))
            model = model.cpu()
            hardprune_f(model, pruning_plan_vgg19_cifar, prune_p_params)
            model = model.to(device)

            cifar_train.initialize(model, lr=2e-5)
            msglogger.info("Number of parameters now: {}".format(num_of_trainable_params(model)))


if __name__ == '__main__':
    main()





