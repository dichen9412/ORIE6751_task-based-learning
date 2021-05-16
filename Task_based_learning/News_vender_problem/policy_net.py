#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import operator
from functools import reduce

import batch
from constants import *

import numpy as np

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

def run_policy_net(X, Y, X_test, Y_test, params, is_nonlinear=False):
    print("running policy net", X.shape, Y.shape)
    if is_nonlinear:
        # Non-linear model, use ADAM step size 1e-3
        # layer_sizes = [params['n'], 200, 200, 1]
        # layers = reduce(operator.add, [[nn.Linear(a,b), nn.BatchNorm1d(b),
        #                                 nn.ReLU(), nn.Dropout(p=0.2)]   # TODO: Why is this 0.2? (others are 0.5)
        #                   for a,b in zip(layer_sizes[0:-2], layer_sizes[1:-1])])
        # layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]
        # model = nn.Sequential(*layers)

        model = nn.Sequential(
                nn.Linear(params['n'], 512, bias=True),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Linear(512, 512, bias=True),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Linear(512, 1, bias=True),
            )
        step_size = 1e-3
    else:
        # Linear model, use ADAM step size 1e-2
        model = nn.Sequential(
            nn.Linear(params['n'], 1)
        )
        step_size = 1e-2

    if USE_GPU:
        model = model.cuda()

    th_frac = 0.8
    inds = np.random.permutation(X.shape[0])
    train_inds = inds[:int(X.shape[0]*th_frac)]
    hold_inds =  inds[int(X.shape[0]*th_frac):]
    X_train, X_hold = X[train_inds, :], X[hold_inds, :]
    Y_train, Y_hold = Y[train_inds, :], Y[hold_inds, :]

    X_train_t = torch.tensor(X_train, dtype=torch.float, device=DEVICE)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float, device=DEVICE)
    X_hold_t = torch.tensor(X_hold, dtype=torch.float, device=DEVICE)
    Y_hold_t = torch.tensor(Y_hold, dtype=torch.float, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float, device=DEVICE)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float, device=DEVICE)

    Y_train_int_t = torch.where(Y_train_t)[1].detach()
    Y_hold_int_t = torch.where(Y_hold_t)[1].detach()
    Y_test_int_t = torch.where(Y_test_t)[1].detach()

    d_ = torch.tensor(params['d'], dtype=torch.float, device=DEVICE)

    # Expected inventory cost
    cost = lambda Z, Y : (params['c_lin'] * Z + 0.5 * params['c_quad'] * (Z**2) +
                      params['b_lin'] * (Y.mv(d_).view(-1,1)-Z).clamp(min=0) +
                      0.5 * params['b_quad'] * (Y.mv(d_).view(-1,1)-Z).clamp(min=0)**2 +
                      params['h_lin'] * (Z-Y.mv(d_).view(-1,1)).clamp(min=0) +
                      0.5 * params['h_quad'] * (Z-Y.mv(d_).view(-1,1)).clamp(min=0)**2) \
                    .mean()

    opt = optim.Adam(model.parameters(), lr=step_size)


    hold_costs, test_costs = [], []
    num_stop_rounds = 20
    last_best = 0

    for i in range(1000):
        model.train()
        batch_size = min(100, X_train_t.shape[0]//5)

        train_cost = batch_train(batch_size, i, X_train_t, Y_train_t, model, opt, cost)

        model.eval()
        test_cost = batch.get_cost(100, i, model, X_test_t, Y_test_t, cost, title="TEST COST")

        hold_cost = batch.get_cost(100, i, model, X_hold_t, Y_hold_t, cost, title="HOLD COST")

        print(i, train_cost.item(), test_cost.item(), hold_cost.item())

        # Early stopping
        test_costs.append(test_cost.item())
        hold_costs.append(hold_cost.item())

        if (hold_cost.item() < hold_costs[last_best]):
            print("new best_hold_cost", hold_cost.item())
            last_best = i

        if batch_size * i > 400 and abs(i - last_best) > 10:
            idx = hold_costs.index(min(hold_costs))
            print("Final test_task_loss:", test_costs[idx])
            return(test_costs[idx])

    # In case of no early stopping, return best run so far
    idx = hold_costs.index(min(hold_costs))
    return test_costs[idx]

    # for i in range(1000):
    #
    #     model.eval()
    #     test_cost = batch.get_cost(100, i, model, X_test_t, Y_test_t, cost)
    #
    #     model.train()
    #     train_cost = batch_train(150, i, X_train_t, Y_train_t, model, opt, cost)
    #
    #     print(train_cost.item(), test_cost.item())
    #
    # return test_cost.item()


def batch_train(batch_sz, epoch, X_train_t, Y_train_t, model, opt, cost_fn):
    train_cost = 0
    batch_data_, batch_targets_ = \
        batch.get_vars(batch_sz, X_train_t, Y_train_t)
    size = batch_sz

    for i in range(0, X_train_t.size(0), batch_sz):

        # Deal with potentially incomplete (last) batch
        if i + batch_sz  > X_train_t.size(0):
            size = X_train_t.size(0) - i
            batch_data_, batch_targets_ = \
                batch.get_vars(size, X_train_t, Y_train_t)

        batch_data_.data[:] = X_train_t[i:i+size]
        batch_targets_.data[:] = Y_train_t[i:i+size]

        opt.zero_grad()
        preds = model(batch_data_)
        batch_cost = cost_fn(preds, batch_targets_)
        batch_cost.backward()
        opt.step()

        ## Keep running average of loss
        train_cost += (batch_cost - train_cost) * size / (i + size)

        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
            epoch, i+size, X_train_t.size(0),
            float(i+size)/X_train_t.size(0)*100,
            batch_cost.item()))

    return train_cost
