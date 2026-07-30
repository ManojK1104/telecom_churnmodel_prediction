import numpy as np
import pandas as pd

def data_ingestion():

    df=pd.read_csv(r"https://raw.githubusercontent.com/ManojK1104/telecom_churnmodel_prediction/refs/heads/main/data/churn.csv")

    return df