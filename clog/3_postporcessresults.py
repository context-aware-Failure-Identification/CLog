import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

TIME_INTERVAL = "180s"
n_clusters = 30


def convert_string_float(stri):
    return np.sum([float(x) for x in stri[1:-1].replace("\n", "").rsplit()])

def calculate_NMI(data):
    z = []
    for x in data.columns:
        if "pred_" in x:
            z.append(x)
    data = data.loc[:, z]
    NMI = []
    for x in z[:-1]:
        NMI.append(normalized_mutual_info_score(data.loc[:, x].values, data.loc[:, z[-1]].values))
    return NMI

def mask_(x):
    if x > 0:
        return 1
    else:
        return 0

# def create_dataset()


# 60s/60s_results.csv
data = pd.read_csv("/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/resources/"+ TIME_INTERVAL +"/results/clustering_holistic_outptu_results_"+TIME_INTERVAL+"_clusters_"+str(n_clusters)+"_.csv")
data["number_error_msgs"] = data.anom_label.apply(lambda x: convert_string_float(x))
data["round_"] = data.round_.apply(lambda x: x[1:-1].replace("\n", "").rsplit()[-1])
# data["round_"] =





data_round_1 = data[data.round_=="1"]
data_round_2 = data[data.round_ == "2"]

def sequential_dataset(data_round_1, round_):
    tests_1 = {}
    test_id_groups = data_round_1.groupby(["test_id"]).groups

    for test_id in test_id_groups.keys():
        z = data_round_1.loc[test_id_groups[test_id]]
        # print(z.encoded_labels)
        test_id_time_seq = z.groupby(["start_time_sequence"]).groups
        for time_int in test_id_time_seq.keys():
            if round_==1:
                tmp = z.loc[test_id_time_seq[time_int]].loc[:, ["sequence_KPI",
                                                                "pred_kmeans",
                                                                "clog_sequence_parent_service",
                                                                "app_error1",
                                                                "assertion_error_round_1",
                                                                "round_1_final_status",
                                                                "Content",
                                                                "anom_label",
                                                                "parent_service",
                                                                "encoded_labels"]]
                tmp.encoded_labels = tmp.encoded_labels.apply(
                    lambda x: " ".join([j for j in x[1:-1].replace("\n", "").replace(" ", "").rsplit(",")]))
                seqs = " "
                for q in np.hstack(tmp.encoded_labels.values):
                    for p in q.rsplit():
                        seqs = seqs + p + " "

                tests_1[str(test_id) + "_test_id_" + time_int] = (tmp.pred_kmeans.values,
                                                                  tmp.sequence_KPI.values.sum(),
                                                                  " ".join([x for x in tmp.clog_sequence_parent_service.values]),
                                                                  " ".join([x for x in tmp.app_error1.values[0].replace("['","").replace("']", "").rsplit()]),
                                                                  " ".join([x for x in tmp.assertion_error_round_1.values[0].replace("['", "").replace("']","").rsplit()]),
                                                                  " ".join([x for x in tmp.round_1_final_status.values[0].replace("['", "").replace("']", "").rsplit()]),
                                                                  " ".join(x for x in tmp.Content.values),
                                                                  tmp.anom_label.values.sum(),
                                                                  tmp.parent_service.values,
                                                                  seqs)
            else:
                tmp = z.loc[test_id_time_seq[time_int]].loc[:,
                      ["sequence_KPI", "pred_kmeans", "clog_sequence_parent_service", "app_error2", "assertion_error_round_2", "round_2_final_status",
                       "anom_label", "Content", "parent_service", "encoded_labels"]]
                tmp.encoded_labels = tmp.encoded_labels.apply(
                    lambda x: " ".join([j for j in x[1:-1].replace("\n", "").replace(" ", "").rsplit(",")]))
                seqs = " "
                for q in np.hstack(tmp.encoded_labels.values):
                    for p in q.rsplit():
                        seqs = seqs + p + " "

                tests_1[str(test_id) + "_test_id_" + time_int] = (tmp.pred_kmeans.values,
                                                                  tmp.sequence_KPI.values.sum(),
                                                                  " ".join([x for x in
                                                                            tmp.clog_sequence_parent_service.values[
                                                                                0].replace("['", "").replace("']",
                                                                                                             "").rsplit()]),
                                                                  " ".join([x for x in tmp.app_error2.values[0].replace("['","").replace("']", "").rsplit()]),
                                                                  " ".join([x for x in tmp.assertion_error_round_2.values[0].replace("['", "").replace("']","").rsplit()]),
                                                                  " ".join([x for x in tmp.round_2_final_status.values[0].replace("['", "").replace("']", "").rsplit()]),
                                                                  " ".join(x for x in tmp.Content.values),
                                                                  tmp.anom_label.values.sum(),
                                                                  tmp.parent_service.values,
                                                                  seqs)

    data_round_1 = pd.DataFrame(tests_1).T
    data_round_1 = data_round_1.reset_index()
    data_round_1.columns = ["ID", "cluster_sequences", "sequence_KPI", "cluster_sequences_parent_service", "app_error1", "assertion_error_round_1", "round_1_final_status", "Content", "anom_label", "parent_service", "encoded_labels"]
    data_round_1["test_id"] = data_round_1.ID.apply(lambda x: float(x.rsplit("_")[0]))
    data_round_1["round_"] = np.full((data_round_1.shape[0],), fill_value=str(round_))
    data_round_1["time"] = data_round_1.ID.apply(lambda x: x.rsplit("_")[-1])
    data_round_1["classification_mask"] = data_round_1.sequence_KPI.apply(lambda x: mask_(x))


    data_round_1["cls_target_api"] = data_round_1.classification_mask*data_round_1.app_error1
    data_round_1["cls_target_assertion"] = data_round_1.classification_mask*data_round_1.assertion_error_round_1
    data_round_1["cls_target_failure"] = data_round_1.classification_mask*data_round_1.round_1_final_status

    round_1_data_clusters = np.zeros((data_round_1.shape[0], n_clusters))

    for x in range(data_round_1.cluster_sequences.values.shape[0]):
        for j in data_round_1.cluster_sequences.values[x]:
            round_1_data_clusters[x, j] += 1


    target = data_round_1.cls_target_failure

    final_data = pd.DataFrame(round_1_data_clusters)
    final_data["target"] = target
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    # vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data_round_1.encoded_labels.values)
    tf_idf_data = pd.DataFrame(X.toarray())
    tf_idf_data.columns = np.array(vectorizer.get_feature_names())
    tf_idf_data["target"] = target
    tf_idf_data["ID"] = data_round_1.ID + "_" + data_round_1.round_

    return final_data, tf_idf_data, data_round_1

def create_sequence_of_clusters(data_round_1):
    d = {}
    for test_id in np.unique(data_round_1.test_id):
        tmp = data_round_1[data_round_1.test_id==test_id]
        seq = np.hstack(data_round_1[data_round_1.test_id == test_id].cluster_sequences.values)
        seq1 = np.hstack(data_round_1[data_round_1.test_id == test_id].cluster_sequences_parent_service.values)
        d[test_id] = (test_id, seq, seq1, tmp.app_error1.iloc[0], tmp.assertion_error_round_1.iloc[0], tmp.round_1_final_status.iloc[0], " ".join(x for x in tmp.Content.values), tmp.sequence_KPI.sum())

    d = pd.DataFrame(d).T
    d.columns = ["test_id", "sequence", "cluster_sequences_parent_service", "app_error", "assesrtion_error", "final_status", "Content", "number_error_msgs"]
    return d


data_round_1["clog_sequence_parent_service"] = data_round_1.pred_kmeans.apply(lambda x: str(x)) + data_round_1.parent_service
data_round_2["clog_sequence_parent_service"] = data_round_2.pred_kmeans.apply(lambda x: str(x)) + data_round_2.parent_service

CLog_data_1, tf_idf_data_1, data_round_1 = sequential_dataset(data_round_1, round_=1)
CLog_data_2, tf_idf_data_2, data_round_2 = sequential_dataset(data_round_2, round_=2)
#
CLog_data = pd.concat([CLog_data_1, CLog_data_2], axis=0)
tf_idf_data = pd.concat([tf_idf_data_1, tf_idf_data_2], axis=0)
tf_idf_data = tf_idf_data.fillna(0)

cluster_seq_round1 = create_sequence_of_clusters(data_round_1)
cluster_seq_round1['round'] = np.full((cluster_seq_round1.shape[0]), fill_value=1)
cluster_seq_round2 = create_sequence_of_clusters(data_round_2)
cluster_seq_round2['round'] = np.full((cluster_seq_round2.shape[0]), fill_value=2)


cluster_seq = pd.concat([cluster_seq_round1, cluster_seq_round2], axis=0)
cluster_seq.to_csv("/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/resources/"+ TIME_INTERVAL +"/HMM_sequencies/sequential_data"+TIME_INTERVAL+"_clusters_"+str(n_clusters)+"_.csv")
cluster_seq_round1.to_csv("/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/resources/"+ TIME_INTERVAL +"/HMM_sequencies/sequential_data_round1_"+TIME_INTERVAL+"_clusters_"+str(n_clusters)+"_.csv")
cluster_seq_round2.to_csv("/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/resources/"+ TIME_INTERVAL +"/HMM_sequencies/sequential_data_round2_"+TIME_INTERVAL+"_clusters_"+str(n_clusters)+"_.csv")

CLog_data.to_csv("/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/resources/"+ TIME_INTERVAL +"/classification_data/classification_CLog_"+TIME_INTERVAL+"_clusters_"+str(n_clusters)+"_.csv")
tf_idf_data.to_csv("/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/resources/"+ TIME_INTERVAL +"/classification_data/classification_TFIDF_"+TIME_INTERVAL+"_.csv")
#
#
data_round_s = pd.concat([data_round_1, data_round_2], axis=0)
data_round_s.to_csv("/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/resources/"+ TIME_INTERVAL +"/HMM_sequencies/novel_seq_"+TIME_INTERVAL+"_clusters_" + str(n_clusters) + "_individual_time_window_.csv")


dataset = []

for idx in np.unique(data_round_1.test_id):
    dataset.append(("".join(x for x in data_round_1[data_round_1.test_id==idx].encoded_labels.values),
                    data_round_1[data_round_1.test_id==idx].sequence_KPI.sum(), str(int(idx))+"_testID_1"))

for idx in np.unique(data_round_2.test_id):
    dataset.append(("".join(x for x in data_round_2[data_round_2.test_id==idx].encoded_labels.values),
                    data_round_2[data_round_2.test_id==idx].sequence_KPI.sum(), str(int(idx))+"_testID_2"))

one_dataset_ = pd.DataFrame(dataset)
one_dataset_.columns = ["sequences", "target", "ID"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(one_dataset_.sequences.values).toarray()
to_save = pd.DataFrame(X)
to_save.columns = vectorizer.get_feature_names()
to_save["sequences"] = one_dataset_.sequences
to_save["target"] = cluster_seq.final_status.values
to_save["ID"] = one_dataset_.ID
to_save.to_csv("/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/resources/"+ TIME_INTERVAL +"/classification_data/classification_TFIDF_"+TIME_INTERVAL+"_clusters_"+str(n_clusters)+"_.csv", index=False)

#
# X = vectorizer.fit_transform(cluster_seq.sequence.apply(lambda x: " ".join(str(j) for j in x))).toarray()
# to_save = pd.DataFrame(X)
# to_save.columns = vectorizer.get_feature_names()
# to_save["sequences"] = one_dataset_.sequences
# to_save["target"] = cluster_seq.final_status.values
# to_save["ID"] = one_dataset_.ID
# to_save.to_csv("/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/resources/"+ TIME_INTERVAL +"/classification_data/classification_TFIDF_"+TIME_INTERVAL+"_OTHER_DATA_.csv", index=False)