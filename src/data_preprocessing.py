from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocessing(df):
    # dropping duplicates col
    df.drop_duplicates()
    # step 2 fill missing values
    ''' 
    no missing values presesnt in dataset
    '''

    # step 3 seprate x and y
    df['Churn']=df['Churn'].map({'Yes':1,'No':0})

    # step 4 label encoder
    le=LabelEncoder()
    for i in df.select_dtypes(include='object'):
        df[i]=le.fit_transform(df[i])

    # step 5 drop unwanted columns
    df.drop(columns=['customerID'],inplace=True)

    # step 6 seprating x and y 
    X=df.drop(columns = 'Churn')
    y=df['Churn']

    # step 7 train test split
    X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.3,
                                               random_state=1)
    
    sm=SMOTE()
    X_train,y_train=sm.fit_resample(X_train,y_train)

    return X_train,X_test,y_train,y_test
