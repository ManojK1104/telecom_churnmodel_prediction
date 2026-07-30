from sklearn.metrics import accuracy_score,classification_report
import pickle
from sklearn.ensemble import RandomForestClassifier
import os

def model_build(X_train,X_test,y_train,y_test):
    # train model

    model=RandomForestClassifier(random_state=1)
    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)
    report=classification_report(y_test,y_pred)
    print('accuracy:',accuracy)
    print('classification:\n',report)

    # save model to pickle 
    os.makedirs("models",exist_ok=True)
    with open("models/model.pkl","wb") as f:
        pickle.dump(model,f)

    return model,accuracy


