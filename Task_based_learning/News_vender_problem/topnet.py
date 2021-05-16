#!/usr/bin/env python3

import numpy as np
import operator
from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim

from qpth.qp import QPFunction

import batch
from constants import *
import cvxpy as cp
import sys
# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

class SolveNewsvendor(nn.Module):
    """ Solve newsvendor scheduling problem """
    def __init__(self, params, eps=1e-2):
        super(SolveNewsvendor, self).__init__()
        k = len(params['d'])
        self.Q = torch.diag(torch.Tensor(
            [params['c_quad']] + [params['b_quad']]*k + [params['h_quad']]*k))
        self.p = torch.Tensor(
            [params['c_lin']] + [params['b_lin']]*k + [params['h_lin']]*k)
        self.G = torch.cat([
            torch.cat([-torch.ones(k,1), -torch.eye(k), torch.zeros(k,k)], 1),
            torch.cat([torch.ones(k,1), torch.zeros(k,k), -torch.eye(k)], 1),
            -torch.eye(1 + 2*k)], 0)
        self.h = torch.Tensor(
            np.concatenate([-params['d'], params['d'], np.zeros(1+ 2*k)]))
        self.one = torch.Tensor([1])
        self.eps_eye = eps * torch.eye(1 + 2*k).unsqueeze(0)

        if USE_GPU:
            self.Q = self.Q.cuda()
            self.p = self.p.cuda()
            self.G = self.G.cuda()
            self.h = self.h.cuda()
            self.one = self.one.cuda()
            self.eps_eye = self.eps_eye.cuda()

    def forward(self, y):
        nBatch, k = y.size()

        Q_scale = torch.cat([torch.diag(torch.cat(
            [self.one, y[i], y[i]])).unsqueeze(0) for i in range(nBatch)], 0)
        Q = self.Q.unsqueeze(0).expand_as(Q_scale).mul(Q_scale)
        p_scale = torch.cat([torch.ones(nBatch, 1, device=DEVICE), y, y], 1)
        p = self.p.unsqueeze(0).expand_as(p_scale).mul(p_scale)
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        e = torch.DoubleTensor()
        if USE_GPU:
            e = e.cuda()

        out = QPFunction(verbose=False)\
            (Q.double(), p.double(), G.double(), h.double(), e, e).float()
        # print("QP_y", y.shape)
        # print("QP_out", out.shape)
        return out[:,:1]

class TOPNet(nn.Module):
    def __init__(self, input_dim, output_dim, params):
        super(TOPNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.constants = [params['c_lin'] , params['c_quad'], params['b_lin'], params['b_quad'], params['h_lin'], params['h_quad']]
        self.constants += list(params['d'])
        self.constants = torch.tensor(self.constants).float()

        self.Pred_Net = nn.Sequential(
                nn.Linear(input_dim, 512, bias=True),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Linear(512, 512, bias=True),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Linear(512, output_dim, bias=True),
                nn.Softmax(dim=1),
            )

        self.Y_Net = nn.Sequential(
                nn.Linear(output_dim, 512, bias=True),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Linear(512, output_dim, bias=True),
            )

        self.Loss_Net = nn.Sequential(
                nn.Linear(output_dim, 512, bias=True),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Linear(512, 512, bias=True),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Linear(512, 1, bias=True),
            )





def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def run_topnet(X, Y, X_test, Y_test, params):

    print("running topnet")
    # Training/validation split
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

    # Expected inventory cost and solver for newsvendor scheduling problem
    cost = lambda Z, Y : (params['c_lin'] * Z + 0.5 * params['c_quad'] * (Z**2) +
                          params['b_lin'] * (Y.mv(d_).view(-1,1)-Z).clamp(min=0) +
                          0.5 * params['b_quad'] * (Y.mv(d_).view(-1,1)-Z).clamp(min=0)**2 +
                          params['h_lin'] * (Z-Y.mv(d_).view(-1,1)).clamp(min=0) +
                          0.5 * params['h_quad'] * (Z-Y.mv(d_).view(-1,1)).clamp(min=0)**2)

    newsvendor_solve = SolveNewsvendor(params)
    if USE_GPU:
        newsvendor_solve = newsvendor_solve.cuda()

    cost_news_fn_batch = lambda x, y: cost(newsvendor_solve(x), y)
    cost_news_fn = lambda x, y: cost(newsvendor_solve(x), y).mean()

    nll = nn.NLLLoss()
    if USE_GPU:
        nll = nll.cuda()

    learning_rate = 0.0001

    model = TOPNet(X_train.shape[1], Y_train.shape[1], params)


    if USE_GPU:
        model = model.cuda()

    opt_pred = optim.Adam(model.Pred_Net.parameters(), lr=learning_rate, weight_decay=1e-4)
    opt_loss = optim.Adam(list(model.Y_Net.parameters()) + list(model.Loss_Net.parameters()), lr=learning_rate, weight_decay=1e-4)

    # For early stopping
    hold_costs, test_costs = [], []
    num_stop_rounds = 20
    last_best = 0

    for i in range(1000):
        model.train()
        batch_size = min(100, X_train_t.shape[0]//5)

        train_cost, train_nll = batch_train(batch_size, i, X_train_t, Y_train_t, Y_train_int_t,\
         model, cost_news_fn_batch, nll, opt_pred, opt_loss, params, cost)

        model.eval()
        test_cost = batch.get_cost(100, i, model.Pred_Net, X_test_t, Y_test_t, cost_news_fn, title="TEST COST")
        test_nll = batch.get_cost_nll(100, i, model.Pred_Net, X_test_t, Y_test_int_t, nll, title="TEST NLL")

        hold_cost = batch.get_cost(100, i, model.Pred_Net, X_hold_t, Y_hold_t, cost_news_fn, title="HOLD COST")
        hold_nll  = batch.get_cost_nll(100, i, model.Pred_Net, X_hold_t, Y_hold_int_t, nll, title="HOLD NLL")

        print(i, train_cost.item(), train_nll.item(), test_cost.item(), test_nll.item(), hold_cost.item(), hold_nll.item())

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



def newsvendor_opt(params, py):
    z = cp.Variable(1)
    d = params['d']
    f = (params['c_lin']*z + 0.5*params['c_quad']*cp.square(z) +
        py.T @ (params['b_lin'] * cp.pos(d-z) +
                0.5 * params['b_quad'] * cp.square(cp.pos(d-z)) +
                params['h_lin'] * cp.pos(z-d) +
                0.5 * params['h_quad'] * cp.square(cp.pos(z-d)) ))
    fval = cp.Problem(cp.Minimize(f), [z >= 0]).solve()
    return z.value, fval

# Inventory ordering cost given demand realization
def f_obj(z, d, params):
    return (params['c_lin'] * z + 0.5 * params['c_quad'] * (z**2) +
            params['b_lin'] * np.maximum(d-z, 0) +
            0.5 * params['b_quad'] * np.maximum(d-z, 0)**2 +
            params['h_lin'] * np.maximum(z-d, 0) +
            0.5 * params['h_quad'] * np.maximum(z-d, 0)**2)


def batch_train(batch_sz, epoch, X_train_t, Y_train_t, Y_train_int_t, model, cost_fn_news, nll, opt_pred, opt_loss, params, cost):

    train_cost_agg = 0
    train_nll_agg = 0

    batch_data_, batch_targets_ = \
        batch.get_vars(batch_sz, X_train_t, Y_train_t)
    _, batch_targets_int_ = \
        batch.get_vars_scalar_out(batch_sz, X_train_t, Y_train_int_t)
    size = batch_sz

    idx = np.arange(X_train_t.size(0))
    np.random.shuffle(idx)

    X_train_t = X_train_t[idx]
    Y_train_t = Y_train_t[idx]
    Y_train_int_t = Y_train_int_t[idx]


    for i in range(0, X_train_t.size(0), batch_sz):

        # Deal with potentially incomplete (last) batch
        if i + batch_sz  > X_train_t.size(0):
            size = X_train_t.size(0) - i
            batch_data_, batch_targets_ = batch.get_vars(
                size, X_train_t, Y_train_t)
            _, batch_targets_int_ = batch.get_vars_scalar_out(
                size, X_train_t, Y_train_int_t)

        batch_data_.data[:] = X_train_t[i:i+size]
        batch_targets_.data[:] = Y_train_t[i:i+size]
        batch_targets_int_.data[:] = Y_train_int_t[i:i+size]


        #set_requires_grad(model.Pred_Net, True)
        preds = model.Pred_Net(batch_data_)
        train_nll  = nll(torch.log(preds), batch_targets_int_)


        y_emb = model.Y_Net(batch_targets_)
        c = model.constants.view(1, -1).repeat(y_emb.shape[0], 1)
        if USE_GPU:
            c = c.cuda()

        est_task_loss = cost(model.Loss_Net(preds), batch_targets_)


        # true_task_cost = []
        # preds_np = preds.cpu().data.numpy()
        # Y = batch_targets_.cpu().data.numpy()
        # for j in range(len(preds_np)):
        #     py = preds_np[j]
        #     z_buy, _ = newsvendor_opt(params, py)
        #     f_eval = f_obj(z_buy, params['d'].dot(Y[j]), params)
        #     true_task_cost.append(f_eval)
        #
        # true_task_cost = torch.tensor(true_task_cost).float()
        # if USE_GPU:
        #     true_task_cost = true_task_cost.cuda()

        true_task_cost = cost_fn_news(preds, batch_targets_).detach()
        train_cost = true_task_cost.mean()


        diff_loss = torch.mean(torch.abs(true_task_cost - est_task_loss) * 0.01)

        if (epoch * (X_train_t.size(0)//batch_sz) <= 400 or diff_loss.item() > 0.5):
            loss = train_nll
        else:
            print("task-loss!")
            loss = est_task_loss.mean()

        opt_pred.zero_grad()
        loss.backward(retain_graph=True)
        opt_pred.step()

        #set_requires_grad(model.Pred_Net, False)
        preds = model.Pred_Net(batch_data_)
        pred_loss = cost(model.Loss_Net(preds.detach()), batch_targets_)
        #model.Loss_Net(torch.cat([preds.detach(), y_emb, c], dim=1))



        diff_loss = torch.mean(torch.abs(true_task_cost- pred_loss) * 0.01)
        loss_net_loss = diff_loss + pred_loss.mean()

        opt_loss.zero_grad()
        loss_net_loss.backward()
        opt_loss.step()

        # Keep running average of losses
        train_cost_agg += (train_cost - train_cost_agg) * batch_sz / (i + batch_sz)
        train_nll_agg += (train_nll - train_nll_agg) * batch_sz / (i + batch_sz)

        print('Epoch: {} [{}/{} ({:.0f}%)]\ttask_Loss: {:.4f}  diff_loss: {:.4f}'.format(
            epoch, i+batch_sz, X_train_t.size(0),
            float(i+batch_sz)/X_train_t.size(0)*100,
            train_cost.item(), diff_loss.item()))
        sys.stdout.flush()
    return train_cost_agg, train_nll_agg
