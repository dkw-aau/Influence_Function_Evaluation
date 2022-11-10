#! /usr/bin/env python3

from train_influence_functions import load_model, load_data
from calc_influence_function import *
from utils import *
if __name__ == "__main__":
    model = load_model()
    trainloader, testloader = load_data()
    test_id_num = 45
    gpu = -1
    recursion_depth = 500
    r = 10
    influences, harmful, helpful =calc_influence_single(model, train_loader, test_loader, test_id_num, gpu,
                          recursion_depth, r, s_test_vec=None,
                          time_logging=False)

