import numpy as np
import pickle
import pandas as pd

TIME_INTERVAL = "60s"
with open("/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/resources/extracted_sequences_"+TIME_INTERVAL+".pickle", "rb") as file:
    data = pickle.load(file)


pom_  = []
for key in data.keys():
    z = pd.DataFrame(data[key])
    z["name"] = np.full((z.shape[0],), fill_value=key)
    z["test_id"] = np.full((z.shape[0],), fill_value=key.rsplit("_")[0])
    z["parent_service"] = np.full((z.shape[0],), fill_value=key.rsplit("_")[1])
    z["roun"] = np.full((z.shape[0],), fill_value=key.rsplit("_")[2])
    pom_.append(z)

data_df = pd.concat(pom_, axis=0)
data_df = data_df.reset_index().drop(["index"], axis=1)
data_df.columns = [ "Content",
                    "level",
               "service",
               "round_1",
               "round_2",
               "api_round_1",
               "api_round_2",
               "assertions_round_1",
               "assertions_round_2",
               "clusters",
               "round",
               "anom_label",
               "encoded_labels",
               "time_hour_day",
                    "name",
                    "test_id",
                    "parent_service",
                    "roun"]

zz = data_df
zz["lenf"] = zz.encoded_labels.apply(lambda x: len(x)).values