from src.data_ingestion import data_ingestion
from src.data_preprocessing import preprocessing
from src.model_building import model_build

def main():
    df=data_ingestion()
    print(df.shape)

    X_train,X_test,y_train,y_test=preprocessing(df)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    model,accuracy=model_build(X_train,X_test,y_train,y_test)
    return model,accuracy


main()