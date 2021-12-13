import torch
import torch.nn as nn

import numpy as np

from networks import *
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from trainer import create_mask

def make_model(src_vocab, tgt_vocab, N=3, d_model=512, d_ff=2048, h=8, dropout=0.1, max_len=20, n_prototypes=10,
               initial_weights=[]):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, max_len)
    kmeans = KMeans(n_prototypes, d_model)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        #         nn.Sequential(nn.Embedding.from_pretrained(initial_weights, freeze=False), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),  #### WHY THIS + 1 it is not clear now 05.07.2021
        number_prototypes=n_prototypes,
        d_model=d_model
    )
    # print("I WAS HERE!!!")
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def investigate_messages_OpenStack(logs_parsed, log_templates):
    """
    Use this script to investigate the templates. Usefull for labeling.

    :param logs_parsed: DataFrame Processed Raw output from Drain
    :param log_templates: DataFrame Templates output from drain
    :return: None
    """
    qwe = []
    for x in range(log_templates.shape[0]):
        print("$$$$$$$" * 10)
        try:
            #         print(np.unique(logs_parsed[logs_parsed.EventId==log_templates.iloc[x, 0]].round_1))
            print(np.unique(logs_parsed[logs_parsed.EventId == log_templates.iloc[x, 0]].round_1))
            if 'NO_FAILURE' not in np.unique(logs_parsed[logs_parsed.EventId == log_templates.iloc[x, 0]].round_1):
                print(np.unique(logs_parsed[logs_parsed.EventId == log_templates.iloc[x, 0]].round_1))
                if len(np.unique(logs_parsed[logs_parsed.EventId == log_templates.iloc[x, 0]].round_1)) == 0:
                    target = "Not Applicable"
                if len(np.unique(logs_parsed[logs_parsed.EventId == log_templates.iloc[x, 0]].round_1)) == 1:
                    target = logs_parsed[logs_parsed.EventId == log_templates.iloc[x, 0]].round_1.iloc[0]
                    qwe.append(log_templates.iloc[x, 0])
        except:
            target = "Not Applicable"
        print("{}\n------------------\n{}\n---------------------\n{}".format(target, log_templates.iloc[x, 0],
                                                                             log_templates.iloc[x, -2]))
        print("$$$$$$$" * 10)


def create_pretained_vector_embeddings_templates(le, log_templates, number_special_tokens):
    """
    :param le: Initialized Label Encoder
    :param log_templates: the templates as generated from
    :param number_special_tokens: how many tokens are you adding (DEFAULT 3) [PAD] [MASK] [CLS]
    :return: Returns modified the log_templates;
    """
    log_templates['labels'] = le.transform(log_templates.EventId) + number_special_tokens
    return log_templates

def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat

class batch_KMeans(object):
    def __init__(self, n_features, n_clusters, n_jobs=4):
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.clusters = np.zeros((self.n_clusters, self.n_features))
        self.count = 1000 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = n_jobs

    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(delayed(_parallel_compute_distance)(X, self.clusters[i]) for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)
        return dis_mat

    def init_cluster(self, X, X_test, indices=None):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_clusters, n_init=20)
        model.fit(X)
        self.kmeans_preds = model.predict(X_test)
        self.clusters = model.cluster_centers_  # copy clusters

    def update_cluster(self, X, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """  # updates the centroid of the cluster
        n_samples = X.shape[0]
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.clusters[cluster_idx] + eta * X[i])
            self.clusters[cluster_idx] = updated_cluster

    def update_assign(self, X):
        dis_mat = self._compute_dist(X)
        return np.argmin(dis_mat, axis=1)

class CA(nn.Module):

    def __init__(self,
                 input_log_events,
                 output_size,
                 d_model,
                 number_layers,
                 number_heads,
                 dropout,
                 max_len,
                 n_clusters,
                 beta,
                 lambda_,
                 optimizer,
                 device,
                 learning_rate,
                 adam_betas_b1,
                 adam_betas_b2,
                 weight_decay,
                 criterion,
                 initial_weights_embeddings):
        super(CA, self).__init__()
        self.beta = beta  # coefficient of the clustering term
        self.lambda_ = lambda_  # coefficient of the reconstruction term
        self.cuda = device
        self.max_len = max_len
        self.n_clusters = n_clusters
        self.n_features = d_model
        self.log_interval = 100
        self.verbose = True
        self.criterion = criterion
        self.weight_decay = weight_decay
        self.weights_initial_embedding = torch.tensor(initial_weights_embeddings, dtype=torch.float32)
        self.PAD_TOKEN = 0
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.predictions = []

        if not self.beta > 0:
            msg = 'beta should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.beta))

        if not self.lambda_ > 0:
            msg = 'lambda should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.lambda_))

        if not self.n_clusters == self.n_clusters:
            msg = '`n_clusters = {} should equal `n_classes = {}`.'
            raise ValueError(msg.format(self.n_clusters, self.n_clusters))


        self.kmeans = batch_KMeans(n_features=self.n_features, n_clusters=self.n_clusters)
        self.Encoder = make_model(src_vocab = input_log_events,
                                  tgt_vocab = output_size,
                                  N=number_layers,
                                  h=number_heads,
                                  d_model=d_model,
                                  d_ff=d_model,
                                  dropout=dropout,
                                  max_len=max_len,
                                  n_prototypes=n_clusters,
                                  initial_weights=self.weights_initial_embedding)
        self.Encoder.cuda()
        self.optimizer = optimizer(self.Encoder.parameters(), lr=learning_rate, betas=(adam_betas_b1, adam_betas_b2),  weight_decay=self.weight_decay)
        self.verbose = True

    def _loss(self, X, cluster_id, label):
        #

        # latent_X = self.Encoder.forward(X.cuda(), None, None, None)
        # next_sentance_prediction = self.Encoder.generator(latent_X)
        # preds = torch.argmax(next_sentance_prediction, axis=1).view(-1, 1)
        #
        # rec_loss = self.criterion(next_sentance_prediction, label.cuda())
        #
        # dist_loss = torch.tensor(0.).to(self.device)
        # clusters = torch.FloatTensor(self.kmeans.clusters).to(self.device)
        #
        # latent_X = latent_X[:, 0, :]
        #
        # for i in range(batch_size):
        #     diff_vec = latent_X[i] - clusters[cluster_id[i]]
        #     sample_dist_loss = torch.matmul(diff_vec.view(1, -1), diff_vec.view(-1, 1))
        #     dist_loss +=  torch.squeeze(sample_dist_loss)
        #
        #
        # loss = self.beta * dist_loss + rec_loss
        # #         a = list(self.parameters())[0].clone()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.Encoder.parameters(), 2.)
        # self.optimizer.step()
        # #         b = list(self.parameters())[0].clone()
        # #         print(torch.equal(a.data, b.data))
        # self.optimizer.zero_grad()
        # dist_loss = self.beta * dist_loss
        #
        # return (rec_loss + dist_loss, rec_loss.detach().cpu().numpy(), dist_loss.detach().cpu().numpy(), preds)
        # return (rec_loss + dist_loss, rec_loss.detach().cpu().numpy(), dist_loss.detach().cpu().numpy())

        batch_size = X.size()[0]

        # with torch.no_grad():

        src_mask = create_mask(X.cuda(), self.PAD_TOKEN)
        latent_X = self.Encoder(X.cuda(), src_mask=src_mask)
        masked_token_pred = self.Encoder.generator.forward(latent_X)

        preds = torch.argmax(masked_token_pred, axis=1).view(-1, 1)
        rec_loss = self.criterion(masked_token_pred.contiguous(), label.cuda().contiguous())

        dist_loss = torch.tensor(0.).to(self.device)
        clusters = torch.FloatTensor(self.kmeans.clusters).to(self.device)

        latent_X = latent_X[:, 0, :]

        for i in range(batch_size):
            diff_vec = latent_X[i] - clusters[cluster_id[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1), diff_vec.view(-1, 1))
            dist_loss += torch.squeeze(sample_dist_loss)

        loss = self.beta * dist_loss + rec_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Encoder.parameters(), 2.)
        self.optimizer.step()
        self.optimizer.zero_grad()

        dist_loss = self.beta * dist_loss

        return (rec_loss + dist_loss, rec_loss.detach().cpu().numpy(), dist_loss.detach().cpu().numpy(), preds)


    def init_clusters_train(self, train_dataloader, test_dataloader, epoch=100, verbose=True):
        batch_X = []
        batch_X_test = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(train_dataloader):
                b_input, label = batch
                src_mask = create_mask(b_input.cuda(), self.PAD_TOKEN)
                latent_X = self.Encoder(b_input.cuda(), src_mask=src_mask)
                latent_X = latent_X[:, 0, :]
                batch_X.append(latent_X.detach().cpu().numpy())

            for batch_idx, batch in enumerate(test_dataloader):
                b_input, label = batch
                src_mask = create_mask(b_input.cuda(), self.PAD_TOKEN)
                latent_X = self.Encoder(b_input.cuda(), src_mask=src_mask)
                latent_X = latent_X[:, 0, :]
                batch_X_test.append(latent_X.detach().cpu().numpy())

        batch_X = np.vstack(batch_X)
        batch_X_test = np.vstack(batch_X_test)
        self.kmeans.init_cluster(batch_X, batch_X_test)


    def _loss_warmup(self, latent_X, label):
        masked_token_pred = self.Encoder.generator.forward(latent_X)
        rec_loss = self.criterion(masked_token_pred.contiguous(), label.cuda().contiguous())
        rec_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Encoder.parameters(), 2.)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return rec_loss


    def forward(self, x):
        out = self.Encoder(x)
        return out


    def fit_warmup(self, train_dataloader, log_epoch, logging_freq):
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            b_input, label = batch
            src_mask = create_mask(b_input.cuda(), self.PAD_TOKEN)
            latent_X = self.Encoder(b_input.cuda(), src_mask=src_mask)
            rec_loss = self._loss_warmup(latent_X, label.cuda())
            total_loss += rec_loss.item()/b_input.size()[0]

        if self.verbose and log_epoch % logging_freq == 0:
            print("-----"*10)
            print(">> Summed Mean MSP loss per batch {} per epoch \t | \t {}".format(log_epoch+1, total_loss))

    def predict_warmup(self, test_dataloader):
        pred_scores = []
        total_loss = 0
        for batch_idx, batch in enumerate(test_dataloader):
            b_input, label = batch
            src_mask = create_mask(b_input.cuda(), self.PAD_TOKEN)
            latent_X = self.Encoder(b_input.cuda(), src_mask=src_mask)
            rec_loss = self._loss_warmup(latent_X, label.cuda())
            total_loss += rec_loss.item() / b_input.size()[0]
            pred_scores.append(self.Encoder.generator(latent_X).detach().cpu().numpy())
        return pred_scores, total_loss


    def fit(self, train_dataloader, epochs):

        self.predictions = []

        for batch_idx, batch in enumerate(train_dataloader):
            b_input, label = batch
            src_mask = create_mask(b_input.cuda(), self.PAD_TOKEN)

            with torch.no_grad():
                latent_X = self.Encoder(b_input.cuda(), src_mask=src_mask)
                latent_X1 = torch.clone(latent_X[:, 0, :])
                latent_X1 = latent_X1.cpu().numpy()



            # [Step-1] Update the assignment results
            cluster_id = self.kmeans.update_assign(latent_X1)
            # [Step-2] Update clusters in batch Kmeans
            elem_count = np.bincount(cluster_id, minlength=self.n_clusters)

            for k in range(self.n_clusters):
                # avoid empty slicing
                if elem_count[k] == 0:
                    continue
                self.kmeans.update_cluster(latent_X1[cluster_id == k], k)

            # [Step-3] Update the network parameters
            loss, rec_loss, dist_loss, preds = self._loss(b_input, cluster_id, label)
            self.predictions += preds


            if self.verbose and batch_idx % self.log_interval == 0:
                #                 msg = 'Epoch: {:02d} | Batch: {:03d} | Loss: {:.3f} | Rec-Loss: {:.3f} | Dist-Loss: {:.3f}'
                #                 print(msg.format(epochs, batch_idx, loss.detach().cpu().numpy()[0], rec_loss.detach().cpu().numpy()[0], dist_loss.detach().cpu().numpy()[0]))

                #                 print(type(rec_loss))
                msg = 'Epoch: {} | Batch: {:03d} | Loss: {} | Loss_rec: {}  | Dist-Loss: {}'
                print(msg.format(epochs, batch_idx, loss.detach().cpu().numpy(), rec_loss,  dist_loss))
