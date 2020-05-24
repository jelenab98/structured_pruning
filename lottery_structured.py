import torch
from cifar_train import CifarTrain
import time
import sys
import json
import os
import pdb
import logging
from datetime import datetime 
from utils import num_of_trainable_params, get_pretrained_vgg19, hardprune_f

from pruning_plans import PruningPlanLottery

log_time = time.strftime("%Y%m%d-%H%M%S")
if not os.path.isdir('./logs'):
    os.mkdir('./logs')

if not os.path.isdir('./logs/' + log_time):
    os.mkdir('./logs/' + log_time)

logging.basicConfig(filename="logs/" + log_time + "/out.log", level=logging.NOTSET)
stdout_handler = logging.StreamHandler(sys.stdout)
msglogger = logging.getLogger()
msglogger.addHandler(stdout_handler)

def lr_decay(lr):
    return lr / 2.0


def main():
    # init train class
    batch_sz = 2
    prune_p_params = 0.05
    model_path = '/home/smocilac/hard_prune/_untrained_vgg19_cifar.pth.tar'
    cifar_train = CifarTrain(log_time, batch_sz)

    # select train and test device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_pretrained_vgg19(model_path).to(device)
    model = model.cpu()
    ppLottery = PruningPlanLottery(model, 0.05)
    model = model.to(device)


    num_params_beg = num_of_trainable_params(model)

    results = {}
    
    for lottery_iter in range(0, 20):
        # reset learning rate
        lr = 0.0025 
        
        num_params_now = num_of_trainable_params(model)
        msglogger.info("Lottery iteration: {}".format(lottery_iter))
        prctng = num_params_now / num_params_beg * 100.0
        msglogger.info("Number of parameters: {} ; {:.2f}%".format(num_params_now, prctng))
        msglogger.info("LR: {}".format(lr))

        iter_str = str(lottery_iter)
        # train model
        for epoch in range(0, 0):
            
            if (epoch + 1) % 10 == 0: # lr decay every 10 epochs
                lr = lr_decay(lr)
                msglogger.info("LR: {}".format(lr))
                cifar_train.initialize(model, lr=lr)
                
            cifar_train.train(epoch, 50000 // batch_sz)
            acc = cifar_train.test(epoch, 10000 // batch_sz)

            if not iter_str in results.keys():
                results[iter_str] = []
            results[iter_str].append(acc)

        model = model.cpu()
        # update pruning plan after training
        ppLottery.get_next_pruning_plan(model)

        # Prune model
        pp_path = "logs/" + log_time + "/prune_config_" + str(lottery_iter) + ".json"
        ppLottery.save_pp(pp_path)
        # reload untrained model
        model = get_pretrained_vgg19(model_path).to(device)
        model = model.cpu()
        # and prune
        hardprune_f(model, ppLottery.prune_by_pruning_plan, prune_p_params)
        model = model.to(device)
    
    results_path = "logs/" + log_time + "/results.json"
    with open(results_path, 'w') as fp:
        json.dump(results, fp, sort_keys=True, indent=4)
    

if __name__ == '__main__':
    main()





