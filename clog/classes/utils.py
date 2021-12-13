import time
import torch
import numpy as np
import torch.nn as nn

import sys
sys.path.append("classes")

from trainer import create_mask

PAD_TOKEN = 0

def evaluate(model, test_loader):
    cluster_assingments = []
    embeddings = []
    next_sentance_preds = []
    for batch_idx, batch in enumerate(test_loader):
        b_input, label = batch
        src_mask = create_mask(b_input.cuda(), PAD_TOKEN)
        # model.eval()
        with torch.no_grad():
            latent_X1 = model.Encoder(b_input.cuda(), src_mask=src_mask)
            latent_X = latent_X1[:, 0, :]
            latent_X = latent_X.cpu().numpy()

        next_sentance_preds.append(model.Encoder.generator(latent_X1).detach().cpu().numpy())
        # latent_X = latent_X[:, 0, :].detach().cpu().numpy()
        cluster_assingments.append(model.kmeans.update_assign(latent_X))
        embeddings.append(latent_X)
    return cluster_assingments, embeddings, next_sentance_preds

def warm_up_MSP(model, train_dataloader, epochs):
    """
    :param model: The MSP model. Composed of Encoder (Transformer) + linear layer that takes as input embedding of the whole sequence.
    :param train_dataloader: The normal training instances used to learn good parameters for the
    :param epochs: How many epoch to pre-train the model to learn good representations.
    TODO:
    1) Consider adding weights in the loss function. This accunts for the infrequent event.
    :return: Trained MSP model
    """
    for e in range(epochs):
        model.train()
        # print("Current epoch is {}".format(e))
        model.fit_warmup(train_dataloader, log_epoch=e, logging_freq=1)
    return model

def evaluate_MSP(model, test_dataloader):
    '''

    :param model: The trained MSP model
    :param test_dataloader: The validation dataloader. used to access performance
    :return: ordered list of softmax scores for each token during warm-up procedure
    '''
    model.eval()
    msp_preds, msp_test_loss = model.predict_warmup(test_dataloader)
    softmax_score_estimates = nn.functional.softmax(torch.Tensor(np.vstack(msp_preds)), dim=0)
    print("Validation loss is {}".format(msp_test_loss))
    return softmax_score_estimates

# pred, values, ns_train_pred = evaluate(model, test_dataloader)  # evaluation on the test_loader
def solver(model, train_dataloader, test_dataloader, epochs):

    start_init_time = time.time()
    model.init_clusters_train(train_dataloader, test_dataloader, epoch=1)

    print(">>>>>> Time for initialziation {} seconds.".format(time.time()-start_init_time))

    y_pred_per_epoch = []
    y_emeddigns_per_epoch = []
    y_ns_train_pred = []

    for e in range(epochs):
        model.train()
        model.fit(train_dataloader, epochs=e)


        model.eval()
        pred, values, ns_train_pred = evaluate(model, test_dataloader)  # evaluation on the test_loader
        #         print(pred.shape)
        y_pred_per_epoch.append(pred)
        y_emeddigns_per_epoch.append(values)
        y_ns_train_pred.append(ns_train_pred)
        print('\nEpoch: ')
    return y_pred_per_epoch, y_emeddigns_per_epoch, y_ns_train_pred, model

def predict(model, test_dataloader):
    model.eval()
    pred, values, next_sentance_preds = evaluate(model, test_dataloader)  # evaluation on the test_loader
    return pred, values, next_sentance_preds