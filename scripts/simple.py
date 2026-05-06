def ReadData(DT,LB):
    F=open(DT,'r')
    L=F.readlines()
    F.close()
    D=[]
    for s in L:
        dt=s.replace('\n','').split(',')
        ls=[]
        for d in dt:
            ls.append(float(d))
        D.append(ls)
    F=open(LB,'r')
    L=F.readlines()
    F.close()
    LA=[]
    for s in L:
        dt=s.replace('\n','')
        LA.append(int(dt))
    return D,LA

def FilterData(DT,LS):
    F=open(DT,'r')
    L=F.readlines()
    F.close()
    D=[]
    for s in L:
        dt=s.replace('\n','').split(',')
        ls=[]
        for d in range(len(dt)):
            if d in LS:
                ls.append(float(dt[d]))
        D.append(ls)
    return D

def AutoEncoder(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation='sigmoid')(input_layer)
    encoded = Dense(8, activation='sigmoid')(encoded) 
    decoded = Dense(16, activation='sigmoid')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=16,
        shuffle=True,
        validation_data=(X_test, X_test),
        verbose=0
    )
    encoder = Model(inputs=input_layer, outputs=encoded)
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)
    classifier_input = Input(shape=(X_train_encoded.shape[1],))
    x = Dense(64, activation='sigmoid')(classifier_input)
    output = Dense(1, activation='sigmoid')(x)
    classifier = Model(inputs=classifier_input, outputs=output)
    classifier.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    classifier.fit(
        X_train_encoded, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.1,
        verbose=0
    )
    y_pred_prob = classifier.predict(X_test_encoded)
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

X,y=ReadData('pid_data.csv','pid_label.csv')
X=np.array(X)
y=np.array(y)

accuracy=AutoEncoder(X,y)
print(accuracy)

#LS=[22,3,20,15,7,2,0,26,8,23,1]
#X=FilterData('pid_data.csv',LS)

accuracy=AutoEncoder(X,y)
print(accuracy)