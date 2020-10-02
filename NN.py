import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split


df=pd.read_csv('/home/ram/Desktop/R=2.6/training1.csv')
Y=df['del_p'].values
X=df.drop(['del_p','R','Re'], axis=1)


X=X.values

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=.1,random_state=101)


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


pca=PCA()
pca.fit(X_train)
X_train_pca=pca.transform(X_train)


X_test1=pd.read_csv('/home/ram/Desktop/R=2.6/testing_fro_Re.csv')


Actual1=pd.DataFrame(data=X_test1,columns=['del_p'])        # testing Target data from Re classification
Actual1.to_csv('actual1.csv', index=False)

X_test1.drop(['del_p','Re','R'],axis=1,inplace=True)
X_test1=scaler.transform(X_test1.values)
X_test1=pd.DataFrame(data=X_test1,columns=['velocity','dia1','density','viscoscity','dia2'] )


X_testa=pd.DataFrame(data=X_test, columns=['velocity','dia1','density','viscoscity','dia2'])

                        
X_test_pca=pca.transform(X_testa)                      #transforming  testing data_inside training range
X_test1_pca=pca.transform(X_test1)   


X_test1=pd.DataFrame(data=X_test1_pca,columns=['velocity','dia1','density','viscoscity','dia2'] )
X_test1.to_csv('X_test1.csv', index=False)                  # saving training data from Re classification

Y_testa=pd.DataFrame(data=y_test, columns=['del_p'])
Y_testa.to_csv('testing_Y.csv',index=False)                 # saving testing target data from main training file

X_testa=pd.DataFrame(data=X_test_pca, columns=['velocity','dia1','density','viscoscity','dia2'])
X_testa.to_csv('testing_X.csv',index=False)                 # saving testing data from main training data


import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model =Sequential()
model.add(Dense(15,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(1, activation='linear'))
#lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate=1e-1,
#    decay_steps=100,
#    decay_rate=0.95)
opt = keras.optimizers.Adam(learning_rate=0.005)

model.compile(optimizer=opt,loss='mse')


model.fit(x=X_train_pca,y=y_train,validation_data=(X_test_pca,y_test),epochs=500)
from keras.models import model_from_json
model_json=model.to_json()
with open('model.json','w') as json_file:

    json_file.write(model_json)

model.save_weights('model.h5')
print('saved model to disk')




