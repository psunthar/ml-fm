import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
import seaborn as sns

df=pd.read_csv('/home/ram/Desktop/R=2.6/training1.csv')
Y=df['del_p'].values
X=df.drop(['del_p','R','Re'], axis=1)


X=X.values

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=.1,random_state=101)


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_testa=pd.DataFrame(data=X_test, columns=['velocity','dia1','density','viscoscity','dia2'])

X_test1=pd.read_csv('/home/ram/Desktop/R=2.6/testing_fro_Re.csv')
Actual1=pd.DataFrame(data=X_test1,columns=['del_p'])
X_test1.drop(['del_p','Re','R'],axis=1,inplace=True)
X_test1=scaler.transform(X_test1.values)
Actual1.to_csv('actual1.csv', index=False)
X_test1=pd.DataFrame(data=X_test1,columns=['velocity','dia1','density','viscoscity','dia2'] )

X_test1.to_csv('X_test1.csv', index=False)




X_testa.to_csv('testing_X.csv',index=False)

Y_testa=pd.DataFrame(data=y_test, columns=['del_p'])
Y_testa.to_csv('testing_Y.csv',index=False)


import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model =Sequential()
#model.add(Dense(18,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(18,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(1, activation='linear'))
#lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate=1e-1,
#    decay_steps=100,
#    decay_rate=0.95)
opt = keras.optimizers.Adam(learning_rate=0.005)

model.compile(optimizer=opt,loss='mse')


model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=500)
from keras.models import model_from_json
model_json=model.to_json()
with open('model.json','w') as json_file:

    json_file.write(model_json)

model.save_weights('model.h5')
print('saved model to disk')



############################################

#losses=pd.DataFrame(model.history.history)
#losses.plot()
#plt.ylim((100,10000))
#plt.show()

#sol=model.predict(X_test)

#soldf=pd.DataFrame(sol,columns=['Predicted'])
#soldf['Actual']=y_test
#sns.scatterplot(x='Actual',y='Predicted',data=soldf)
#plt.show()
#####################
#f=np.linspace(0,17000,17000)
#plt.plot(f,f)
#plt.show()



####################
#test_s=pd.read_csv('/home/ram/testing_data_S.csv')
#test_g=pd.read_csv('/home/ram/testing_data_G.csv')
#test_s_data=test_s.drop(['Re','del_p','R'],axis=1)
#test_g_data=test_g.drop(['Re','del_p','R'],axis=1)


#print(test_s_data)
#soldf['predicted_s']=model.call(test_s_data)
#soldf['predicted_g']=model.call(test_g_data)
#print(soldf)

#soldf['actual_s']=test_s['del_p']
#soldf['actual_g']=test_g['del_p']

#sns.scatterplot(x='actual_s',y='predicted_s',data=soldf)
#sns.scatterplot(x='actual_p',y='predicted_g',data=soldf)
#plt.show()
#print(soldf)

