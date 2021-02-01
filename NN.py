import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split


df=pd.read_csv('training1.csv')
Y=df['del_p'].values
X=df.drop(['del_p','R'], axis=1)


X=X.values

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=.1,random_state=101)
re_ci=pd.DataFrame(data=X_test,columns=['velocity','dia1','density','viscoscity','dia2','Re','Ci'])
re_ci.drop(['velocity','dia1','density','dia2','viscoscity','Re'], axis=1,inplace=True)
plt.show()
re_ci.to_csv('r_square_test.csv',index=False)
X_train=pd.DataFrame(data=X_train,columns=['velocity','dia1','density','viscoscity','dia2','Re','Ci'])
X_train=X_train.drop(['Re','Ci'],axis=1)
X_train=X_train.to_numpy()
X_test=pd.DataFrame(data=X_test,columns=['velocity','dia1','density','viscoscity','dia2','Re','Ci'])
X_test=X_test.drop(['Re','Ci'],axis=1)
X_test=X_test.to_numpy()

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


pca=PCA(n_components=4)
pca.fit(X_train)
X_train_pca=pca.transform(X_train)


X_test1=pd.read_csv('testing_fro_Re.csv')
X_test1.drop(['Ci'],axis=1,inplace=True)


Y_test1=pd.DataFrame(data=X_test1,columns=['del_p'])        # testing Target data from Re classification
Y_test1.to_csv('Y_test1.csv', index=False)

X_test1.drop(['del_p','Re','R'],axis=1,inplace=True)

X_test1=scaler.transform(X_test1.values)
#X_test1=pd.DataFrame(data=X_test1,columns=['velocity','dia1','density','viscoscity'] )


#X_testa=pd.DataFrame(data=X_test, columns=['velocity','dia1','density','viscoscity'])

                        
X_test_pca=pca.transform(X_test)                      #transforming  testing data_inside training range
X_test1_pca=pca.transform(X_test1)   


X_test1_pca=pd.DataFrame(data=X_test1_pca,columns=['velocity','dia1','density','viscoscity'] )
X_test1_pca.to_csv('X_test1.csv', index=False)                  # saving training data from Re classification


Y_test=pd.DataFrame(data=y_test, columns=['del_p'])
Y_test.to_csv('testing_Y.csv',index=False)                 # saving testing target data from main training file



X_test_pca=pd.DataFrame(data=X_test_pca, columns=['velocity','dia1','density','viscoscity'])
X_test_pca.to_csv('testing_X.csv',index=False)                 # saving testing data from main training data

Y_test_whole=pd.concat([Y_test,Y_test1],axis=0)
X_test_pca_whole=pd.concat([X_test_pca,X_test1_pca],axis=0)    # to get the loss on testing data (outside training range)

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model =Sequential()

model.add(Dense(15,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(1, activation='linear'))

opt = keras.optimizers.Adam(learning_rate=0.005)

model.compile(optimizer=opt,loss='mse')


model.fit(x=X_train_pca,y=y_train,validation_data=(X_test_pca_whole,Y_test_whole),epochs=200)
A=np.log(model.history.history['loss'])
B=np.log(model.history.history['val_loss'])
losses=pd.DataFrame(data=A,columns=['test_loss'])
losses['val_loss']=B
losses.to_csv('losses.csv')

#model.save('model.h5')
from keras.models import model_from_json
model_json=model.to_json()
with open('model.json','w') as json_file:

    json_file.write(model_json)

model.save_weights('model.h5')
print('saved model to disk')




