#!/usr/bin/env python3

import torch
from constants import *

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

def get_vars(batch_sz, X_test_t, Y_test_t):
    batch_data_ = torch.empty(batch_sz, X_test_t.size(1), device=DEVICE)
    batch_targets_ = torch.empty(batch_sz, Y_test_t.size(1), device=DEVICE)

    return batch_data_, batch_targets_

def get_vars_scalar_out(batch_sz, X_test_t, Y_test_t):
    batch_data_ = torch.empty(batch_sz, X_test_t.size(1), device=DEVICE)
    batch_targets_ = torch.empty(batch_sz, dtype=torch.long, device=DEVICE)

    return batch_data_, batch_targets_

# General batch evaluation
def get_cost_helper(batch_sz, epoch, model, X_test_t, Y_test_t,
    loss_fn, var_getter_fn, title):

    test_cost = 0

    batch_data_, batch_targets_ = var_getter_fn(
        batch_sz, X_test_t, Y_test_t)
    size = batch_sz

    for i in range(0, X_test_t.size(0), batch_sz):

        # Deal with potentially incomplete (last) batch
        if i + batch_sz  > X_test_t.size(0):
            size = X_test_t.size(0) - i
            batch_data_, batch_targets_ = var_getter_fn(
                size, X_test_t, Y_test_t)

        batch_data_.data[:] = X_test_t[i:i+size]
        batch_targets_.data[:] = Y_test_t[i:i+size]

        #print("batch_data_", batch_data_.shape)
        preds = model(batch_data_)
        batch_cost = loss_fn(preds, batch_targets_)

        # Keep running average of loss
        test_cost += (batch_cost - test_cost) * size / (i + size)

    print(title + ':' + ' ' * 20)
    print('Average loss: {:.4f}'.format(test_cost.item()))

    return test_cost

def get_cost(batch_sz, epoch, model, X_test_t, Y_test_t, loss_fn, title='TEST SET RESULTS'):
    return get_cost_helper(batch_sz, epoch, model, X_test_t, Y_test_t,
        loss_fn, get_vars, title)

def get_cost_nll(batch_sz, epoch, model, X_test_t, Y_test_t, loss_fn, title='TEST SET RESULTS'):
    return get_cost_helper(batch_sz, epoch, model, X_test_t, Y_test_t,
        loss_fn, get_vars_scalar_out, title)
