
from train_influence_functions import load_model, load_data
from calc_influence_function import *
import csv
if __name__ == "__main__":
    model = load_model()
    trainloader, testloader = load_data()
    test_id_num = 45
    gpu = 1
    recursion_depth = 500
    r = 10
    influences=calc_influence_single(model, trainloader, testloader, test_id_num, gpu,
                          recursion_depth, r, s_test_vec=None,
                          time_logging=False)

    with open(..., 'wb') as influenceimg:
        wr = csv.writer(influenceimg, quoting=csv.QUOTE_ALL)
        wr.writerow(influences)
