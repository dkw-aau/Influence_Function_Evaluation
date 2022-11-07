#! /usr/bin/env python3

from train_influence_functions import load_model, load_data
from calc_influence_function import *


model = load_model()
trainloader, testloader = load_data()
#     ptif.init_logging('logfile.log')
influences, harmful, helpful=calc_influence_single (model, trainloader, testloader,test_id_num=45, gpu=-1, recursion_depth=500, r=10 )