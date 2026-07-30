import numpy as np
import pandas as pd

def data_ingestion():

    df=pd.read_csv(r"E:\telecom_churnmodel_prediction\data\churn.csv")

    return df