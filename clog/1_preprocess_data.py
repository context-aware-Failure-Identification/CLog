import pandas as pd
import numpy as np
from datetime import datetime
import pickle

TIME_INTERVAL_ = "600s"

data = pd.read_csv("/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/NOVA_clusters_processed_padded.csv")

# def mapping_tmp(x):
#     print(x)
#     return

data["time_hour_day"] = data.time_hour.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))

def group_data_test_id(data, test_id):
    return data[data.test_id==test_id]

def group_data_service(data, service):
    return data[data.parent_service == service]

def group_(data):
    test_vals = {}
    for test_id in np.unique(data.test_id.values):
        print("----------"*10)
        print("Start processing test {}".format(test_id))
        for rou in np.unique(data.loc[:, "round"].values):
            data1 = data[data.test_id == test_id]
            data1 = data1[data1.loc[:, "round"].values == rou]
            data1.index = data1.time_hour_day
            grs = data1.groupby([pd.Grouper(key="time_hour_day", freq=TIME_INTERVAL_)]).groups
            for idx in range(len(list(grs.keys()))):
                for service in np.unique(data.parent_service.values):
                    output = []
                    # data1 = data1[data1.parent_service == service]
                    if len(grs[list(grs.keys())[idx]]) > 0:
                        pom = data1.loc[grs[list(grs.keys())[idx]], :]
                        pom = pom[pom.parent_service == service]
                        output.append((pom.Content.values,
                                       pom.level.values,
                                       pom.service.values,
                                       pom.round_1.values,
                                       pom.round_2.values,
                                       pom.api_round_1.values,
                                       pom.api_round_2.values,
                                       pom.assertions_round_1.values,
                                       pom.assertions_round_2.values,
                                       pom.clusters.values,
                                       pom.loc[:, "round"].values,
                                       # data1.loc[grs[list(grs.keys())[idx]], :].EventTemplate.values,
                                       # data1.loc[grs[list(grs.keys())[idx]], :].ParameterList.values,
                                       pom.anom_label.values,
                                       pom.encoded_labels.values,
                                       pom.time_hour_day.values))
                    # print(grs[list(grs.keys())[idx]])
                        test_vals[str(test_id) + "_" + service + "_" + str(rou) + "_" + grs[list(grs.keys())[idx]][0].strftime('%Y-%m-%d %H:%M:%S.%f')] = output
        print("Finish processing test {}".format(test_id))

    return test_vals

processed_data = group_(data)

with open("/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/resources/extracted_sequences_" + TIME_INTERVAL_+ ".pickle", "wb") as file:
    pickle.dump(processed_data, file)