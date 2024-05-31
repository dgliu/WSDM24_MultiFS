import logging
import sys, os
import torch
import pickle
from dataloader import Dtfloader
import modules.models as models

def getModel(model:str, opt):
    model = model.lower()
    if model == "dnn":
        return models.DNN(opt)
    elif model == "deepfm":
        return models.DeepFM(opt)
    elif model == "dcn":
        return models.DeepCrossNet(opt)
    else:
        raise ValueError("Invalid model type: {}".format(model))

def getOptim(network, optim, lr, l2):
    params = network.parameters()
    optim = optim.lower()
    if optim == "sgd":
        return torch.optim.SGD(params, lr= lr, weight_decay = l2)
    elif optim == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay = l2)
    else:
        raise ValueError("Invalid optmizer type:{}".format(optim))

def getDevice(device_id):
    if device_id != -1:
        assert torch.cuda.is_available(), "CUDA is not available"
        # torch.cuda.set_device(device_id)
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def getDataLoader(dataset:str, path):
    dataset = dataset.lower()
    if dataset == 'AliExpress-1':
        return Dtfloader.AliExpressLoader(path)
    elif dataset == 'AliExpress-2':
        return Dtfloader.AliExpressLoader(path)
    elif dataset == 'ali-ccp':
        return Dtfloader.Ali_CCPLoader(path)

def get_stats(path):
    defaults_path = os.path.join(path + "/defaults.pkl")
    with open(defaults_path, 'rb') as fi:
        defaults = pickle.load(fi)
    return [i+1 for i in list(defaults.values())]

def get_log(name=""):
    FORMATTER = logging.Formatter(fmt="[{asctime}]:{message}", style= '{')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(FORMATTER)
    logger.addHandler(ch)
    return logger