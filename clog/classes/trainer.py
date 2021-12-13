import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, model, criterion, opt=None, is_test=False):
        self.model = model
        self.criterion = criterion
        self.opt = opt
        self.is_test = is_test

    def __call__(self, x, y, dist):
        loss = torch.mean((1 - y) * torch.sqrt(dist) - (y) * torch.log(1 - torch.exp(-torch.sqrt(dist))))

        if not self.is_test:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.opt is not None:
                self.opt.step()
                self.opt.zero_grad()

        return loss.item()


def create_mask(b_input, pad):
    return (b_input != pad).unsqueeze(-2)

def run_train(dataloader, model, loss_compute, step_size=10):
    "Standard Training and Logging Function"
    start = time.time()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        b_input, b_labels = batch
        src_mask = create_mask(b_input)
        out = model.forward(b_input.cuda(), b_labels.cuda(), None, None)
        dist = torch.sum((out[:, 0, :] - model.c) ** 2, dim=1)
        loss = loss_compute(out, b_labels.cuda(), dist)
        total_loss += loss
        if i % step_size == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d / %d Loss: %f" %
                  (i, len(dataloader), loss))
            start = time.time()
            tokens = 0
    return total_loss

def run_test(dataloader, model, loss_compute, step_size=10):
    "Standard Training and Logging Function"
    preds = []
    distances = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            b_input, b_labels = batch
            out = model.forward(b_input.cuda(), b_labels.cuda(), None, None)
            out_p = model.generator(out)
            dist = torch.sum((out[:, 0, :] - model.c) ** 2, dim=1)
            loss = loss_compute(out, b_labels.cuda(), dist)
            if i % step_size == 1:
                print("Epoch Step: %d / %d Loss: %f" %(i, len(dataloader), loss))
            tmp = out_p.cpu().numpy()
            preds += list(np.argmax(tmp, axis=1))
            distances += list(dist.cpu().numpy())

    return preds, distances

def run_test_out_vectors(dataloader, model, loss_compute, step_size=10):
    "Standard Training and Logging Function"
    preds = []
    distances = []
    out_vectors = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            b_input, b_labels = batch
            out = model.forward(b_input.cuda(), b_labels.cuda(), None, None)
            out_p = model.generator(out)
            dist = torch.sum((out[:, 0, :] - model.c) ** 2, dim=1)
            loss = loss_compute(out, b_labels.cuda(), dist)
            if i % step_size == 1:
                print("Epoch Step: %d / %d Loss: %f" % (i, len(dataloader), loss))
            tmp = out_p.cpu().numpy()
            preds += list(np.argmax(tmp, axis=1))
            distances += list(dist.cpu().numpy())
            out_vectors += list(out_p[:, 0, :].numpy())
    return preds, distances, out_vectors