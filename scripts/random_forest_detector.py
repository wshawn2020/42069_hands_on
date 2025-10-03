import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestRegressor


def preparation(dataset):
    data_benign = dataset[dataset[' Label']=='BENIGN']

    data_to_append = pd.DataFrame(dataset[dataset[' Label']=='DoS'])
    DoS_df = pd.concat([data_benign, data_to_append], ignore_index=True)

    data_to_append = pd.DataFrame(dataset[dataset[' Label']=='DDoS'])
    DDoS_df = pd.concat([data_benign, data_to_append], ignore_index=True)

    data_to_append = dataset[dataset[' Label']=='PortScan']
    PortScan_df = pd.concat([data_benign, data_to_append], ignore_index=True)

    NA_df = dataset
    NA_df[' Label'] = dataset[' Label'].apply({'DoS':'Anormal','BENIGN':'Normal' ,'DDoS':'Anormal', 'PortScan':'Anormal'}.get)

    return DoS_df, DDoS_df, PortScan_df, NA_df

def feature_selection(df):
    feature = (df.drop([' Label'],axis=1)).columns.values
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])

    X = df.drop([' Label'], axis=1) 
    X[np.isfinite(X) == True] = 0
    Y = df.iloc[:, -1].values.reshape(-1,1)
    Y = np.ravel(Y)

    imputer = Imputer(missing_values=np.nan, strategy = "mean")
    imputer = imputer.fit(X)
    X = imputer.transform(X)
    rf = RandomForestRegressor()
    rf.fit(X, Y)

    print ("Features sorted:")
    print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature), reverse=True))

def train_test_dataset(df):
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    X = df.drop([' Label'],axis=1) 
    y = df.iloc[:, -1].values.reshape(-1,1)
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 0, stratify = y)
    return  X_train, X_test, y_train, y_test

def detect_dataset(df, threat_name):
    # Segment train and test sets of data
    X_train, X_test, y_train, y_test = train_test_dataset(df)

    X_train = X_train.astype('int')
    X_test = X_test.astype('int')
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # Call Random Forest detector
    rf_score, rf_precision, rf_recall, rf_fscore, none = RandomForest(X_train, X_test, y_train, y_test, threat_name)

def RandomForest(X_train, X_test, y_train, y_test, threat_name):
    rf = RandomForestClassifier(random_state = 0)
    imputer = Imputer(missing_values=np.nan, strategy="mean")
    imputer = imputer.fit(X_train)

    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    y_pred = rf.predict(X_test)
    y_true = y_test

    print('Detection Results:')
    precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_pred, average='weighted') 
    print('Accuracy: ' + str(rf_score) + ' | Precision: '+ str(precision) + " | Recall: " + str(recall) + " | Fscore: " + str(fscore))
    
    output_dir = 'results'
    output_fname = 'rf_heatmap_' + threat_name + '.png'
    full_path = os.path.join(output_dir, output_fname)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cm = confusion_matrix(y_true, y_pred)
    f,ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.savefig(full_path)
    print('Heatmap saved to: ' + full_path)

    return rf_score, precision, recall, fscore, none

if __name__ == "__main__":
    # Read dataset
    print("[DATA] Loading dataset")
    dataset = pd.read_csv('data/dataset.csv')

    # Extra dataset preprocessing clean steps (towards NaN and Infinity cases)
    print("[PREP] Cleaning data")
    dataset.dropna(inplace=True)
    indices_to_keep = ~dataset.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    dataset = dataset[indices_to_keep]

    # Data preprocessing for diverse threat categories
    print("[PREP] Prepare subsets for specific threats")
    DoS_df, DDoS_df, PortScan_df, NA_df = preparation(dataset)

    # Features sorted by scores
    print("[RANK] Sort features by scores")
    feature_selection(dataset)

    # Detect threats via Random Forest Detector
    print("[DETECT] Random forest detects DoS")
    detect_dataset(DoS_df, 'DoS')

    print("[DETECT] Random forest detects DDoS")
    detect_dataset(DDoS_df, 'DDoS')

    print("[DETECT] Random forest detects PortScan")
    detect_dataset(PortScan_df, 'PortScan')

    print("[DETECT] Random forest detects diverse threats")
    detect_dataset(NA_df, 'NA')

    print("[FINISH] Work done")