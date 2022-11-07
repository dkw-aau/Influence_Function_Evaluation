import torch
from torch.autograd import grad
from utils import *
import time
import datetime
import numpy as np
import copy
import logging

from pathlib import Path


def s_test(z_test, t_test, model, z_loader, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=5000):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.
    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.
    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(z_test, t_test, model, gpu)
    h_estimate = v.copy()

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO: do x, t really have to be chosen RANDOMLY from the train set?
        #########################
        for x, t in z_loader:
            if gpu >= 0:
                x, t = x.cuda(), t.cuda()
            y = model(x)
            loss = calc_loss(y, t)
            params = [ p for p in model.parameters() if p.requires_grad ]
            hv = hvp(loss, params, h_estimate)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
        display_progress("Calc. s_test recursions: ", i, recursion_depth)
    return h_estimate


def calc_loss(y, t):
    """Calculates the loss
    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)
    Returns:
        loss: scalar, the loss"""
    ####################
    # if dim == [0, 1, 3] then dim=0; else dim=1
    ####################
    # y = torch.nn.functional.log_softmax(y, dim=0)
    y = torch.nn.functional.log_softmax(y)
    loss = torch.nn.functional.nll_loss(
        y, t, weight=None, reduction='mean')
    return loss


def grad_z(z, t, model, gpu=-1):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.
    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU
    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    # initialize
    if gpu >= 0:
        z, t = z.cuda(), t.cuda()
    y = model(z)
    loss = calc_loss(y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    return list(grad(loss, params, create_graph=True))


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.
    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads

def calc_s_test(model, test_loader, train_loader, save=False, gpu=-1,
                damp=0.01, scale=25, recursion_depth=5000, r=1, start=0):
    """Calculates s_test for the whole test dataset taking into account all
    training data images.
    Arguments:
        model: pytorch model, for which s_test should be calculated
        test_loader: pytorch dataloader, which can load the test data
        train_loader: pytorch dataloader, which can load the train data
        save: Path, path where to save the s_test files if desired. Omitting
            this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        start: int, index of the first test index to use. default is 0
    Returns:
        s_tests: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge.
        save: Path, path to the folder where the s_test files were saved to or
            False if they were not saved."""
    if save and not isinstance(save, Path):
        save = Path(save)
    if not save:
        logging.info("ATTENTION: not saving s_test files.")

    s_tests = []
    for i in range(start, len(test_loader.dataset)):
        z_test, t_test = test_loader.dataset[i]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])

        s_test_vec = calc_s_test_single(model, z_test, t_test, train_loader,
                                        gpu, damp, scale, recursion_depth, r)

        if save:
            s_test_vec = [s.cpu() for s in s_test_vec]
            torch.save(
                s_test_vec,
                save.joinpath(f"{i}_recdep{recursion_depth}_r{r}.s_test"))
        else:
            s_tests.append(s_test_vec)
        display_progress(
            "Calc. z_test (s_test): ", i-start, len(test_loader.dataset)-start)

    return s_tests, save


def calc_s_test_single(model, z_test, t_test, train_loader, gpu=-1,
                       damp=0.01, scale=25, recursion_depth=5000, r=1):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))
    Arguments:
        model: pytorch model, for which s_test should be calculated
        z_test: test image
        t_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""
    s_test_vec_list = []
    for i in range(r):
        s_test_vec_list.append(s_test(z_test, t_test, model, train_loader,
                                      gpu=gpu, damp=damp, scale=scale,
                                      recursion_depth=recursion_depth))
        display_progress("Averaging r-times: ", i, r)

    ################################
    # TODO: Understand why the first[0] tensor is the largest with 1675 tensor
    #       entries while all subsequent ones only have 335 entries?
    ################################
    s_test_vec = s_test_vec_list[0]
    for i in range(1, r):
        s_test_vec += s_test_vec_list[i]

    s_test_vec = [i / r for i in s_test_vec]

    return s_test_vec


def calc_grad_z(model, train_loader, save_pth=False, gpu=-1, start=0):
    """Calculates grad_z and can save the output to files. One grad_z should
    be computed for each training data sample.
    Arguments:
        model: pytorch model, for which s_test should be calculated
        train_loader: pytorch dataloader, which can load the train data
        save_pth: Path, path where to save the grad_z files if desired.
            Omitting this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        start: int, index of the first test index to use. default is 0
    Returns:
        grad_zs: list of torch tensors, contains the grad_z tensors
        save_pth: Path, path where grad_z files were saved to or
            False if they were not saved."""
    if save_pth and isinstance(save_pth, str):
        save_pth = Path(save_pth)
    if not save_pth:
        logging.info("ATTENTION: Not saving grad_z files!")

    grad_zs = []
    for i in range(start, len(train_loader.dataset)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        grad_z_vec = grad_z(z, t, model, gpu=gpu)
        if save_pth:
            grad_z_vec = [g.cpu() for g in grad_z_vec]
            torch.save(grad_z_vec, save_pth.joinpath(f"{i}.grad_z"))
        else:
            grad_zs.append(grad_z_vec)
        display_progress(
            "Calc. grad_z: ", i-start, len(train_loader.dataset)-start)

    return grad_zs, save_pth


def load_s_test(s_test_dir=Path("./s_test/"), s_test_id=0, r_sample_size=10,
                train_dataset_size=-1):
    """Loads all s_test data required to calculate the influence function
    and returns a list of it.
    Arguments:
        s_test_dir: Path, folder containing files storing the s_test values
        s_test_id: int, number of the test data sample s_test was calculated
            for
        r_sample_size: int, number of s_tests precalculated
            per test dataset point
        train_dataset_size: int, number of total samples in dataset;
            -1 indicates to use all available grad_z files
    Returns:
        e_s_test: list of torch vectors, contains all e_s_tests for the whole
            dataset.
        s_test: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge."""
    if isinstance(s_test_dir, str):
        s_test_dir = Path(s_test_dir)

    s_test = []
    logging.info(f"Loading s_test from: {s_test_dir} ...")
    num_s_test_files = len(s_test_dir.glob("*.s_test"))
    if num_s_test_files != r_sample_size:
        logging.warn("Load Influence Data: number of s_test sample files"
                     " mismatches the available samples")
    ########################
    # TODO: should prob. not hardcode the file name, use natsort+glob
    ########################
    for i in range(num_s_test_files):
        s_test.append(
            torch.load(s_test_dir / str(s_test_id) + f"_{i}.s_test"))
        display_progress("s_test files loaded: ", i, r_sample_size)

    #########################
    # TODO: figure out/change why here element 0 is chosen by default
    #########################
    e_s_test = s_test[0]
    # Calculate the sum
    for i in range(len(s_test)):
        e_s_test = [i + j for i, j in zip(e_s_test, s_test[0])]

    # Calculate the average
    #########################
    # TODO: figure out over what to calculate the average
    #       should either be r_sample_size OR e_s_test
    #########################
    e_s_test = [i / len(s_test) for i in e_s_test]

    return e_s_test, s_test


def load_grad_z(grad_z_dir=Path("./grad_z/"), train_dataset_size=-1):
    """Loads all grad_z data required to calculate the influence function and
    returns it.
    Arguments:
        grad_z_dir: Path, folder containing files storing the grad_z values
        train_dataset_size: int, number of total samples in dataset;
            -1 indicates to use all available grad_z files
    Returns:
        grad_z_vecs: list of torch tensors, contains the grad_z tensors"""
    if isinstance(grad_z_dir, str):
        grad_z_dir = Path(grad_z_dir)

    grad_z_vecs = []
    logging.info(f"Loading grad_z from: {grad_z_dir} ...")
    available_grad_z_files = len(grad_z_dir.glob("*.grad_z"))
    if available_grad_z_files != train_dataset_size:
        logging.warn("Load Influence Data: number of grad_z files mismatches"
                     " the dataset size")
        if -1 == train_dataset_size:
            train_dataset_size = available_grad_z_files
    for i in range(train_dataset_size):
        grad_z_vecs.append(torch.load(grad_z_dir / str(i) + ".grad_z"))
        display_progress("grad_z files loaded: ", i, train_dataset_size)

    return grad_z_vecs


def calc_influence_function(train_dataset_size, grad_z_vecs=None,
                            e_s_test=None):
    """Calculates the influence function
    Arguments:
        train_dataset_size: int, total train dataset size
        grad_z_vecs: list of torch tensor, containing the gradients
            from model parameters to loss
        e_s_test: list of torch tensor, contains s_test vectors
    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness"""
    if not grad_z_vecs and not e_s_test:
        grad_z_vecs = load_grad_z()
        e_s_test, _ = load_s_test(train_dataset_size=train_dataset_size)

    if (len(grad_z_vecs) != train_dataset_size):
        logging.warn("Training data size and the number of grad_z files are"
                     " inconsistent.")
        train_dataset_size = len(grad_z_vecs)

    influences = []
    for i in range(train_dataset_size):
        tmp_influence = -sum(
            [
                ###################################
                # TODO: verify if computation really needs to be done
                # on the CPU or if GPU would work, too
                ###################################
                torch.sum(k * j).data.cpu().numpy()
                for k, j in zip(grad_z_vecs[i], e_s_test)
                ###################################
                # Originally with [i] because each grad_z contained
                # a list of tensors as long as e_s_test list
                # There is one grad_z per training data sample
                ###################################
            ]) / train_dataset_size
        influences.append(tmp_influence)
        display_progress("Calc. influence function: ", i, train_dataset_size)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist()


def calc_influence_single(model, train_loader, test_loader, test_id_num, gpu,
                          recursion_depth, r, s_test_vec=None,
                          time_logging=False):
    """Calculates the influences of all training data points on a single
    test dataset image.
    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated
    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""
    # Calculate s_test vectors if not provided
    if not s_test_vec:
        z_test, t_test = test_loader.dataset[test_id_num]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])
        s_test_vec = calc_s_test_single(model, z_test, t_test, train_loader,
                                        gpu, recursion_depth=recursion_depth,
                                        r=r)

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in range(train_dataset_size):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        if time_logging:
            time_a = datetime.datetime.now()
        grad_z_vec = grad_z(z, t, model, gpu=gpu)
        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(f"Time for grad_z iter:"
                         f" {time_delta.total_seconds() * 1000}")
        tmp_influence = -sum(
            [
                ####################
                # TODO: potential bottle neck, takes 17% execution time
                # torch.sum(k * j).data.cpu().numpy()
                ####################
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size
        influences.append(tmp_influence)
        display_progress("Calc. influence function: ", i, train_dataset_size)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist()

