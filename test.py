##############################################  TESTING FILE  ###################################

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import keras
from keras.models import model_from_json
from sklearn.metrics import r2_score
#loaded_model=load_model('model.h5')


json_file=open('model.json','r')
loaded_model_json= json_file.read()
json_file.close()
loaded_model=tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
print('loaded model_from disk')


opt = keras.optimizers.Adam(learning_rate=0.005)
loaded_model.compile(optimizer=opt,loss='mse')
X_test=pd.read_csv('testing_X.csv')
X_test1=pd.read_csv('X_test1.csv')              # testing data split based on Re

#Actual1=pd.DataFrame(data=X_test1,columns=['del_p'])

#X_test1.drop(['del_p','Re','R'],axis=1,inplace=True)


sol=loaded_model.predict(X_test.values)         # in range
sol1=loaded_model.predict(X_test1.values)       # outside range

########################Evaluation metrics##################################
Y_test1=pd.read_csv('Y_test1.csv')
Y_test=pd.read_csv('testing_Y.csv')

print(r2_score(Y_test,sol))
print(r2_score(Y_test1,sol1))



sol=pd.DataFrame(data=sol, columns=['predict'])
sol1=pd.DataFrame(data=sol1, columns=['predict'])
print(sol1)

#Actual1.to_csv('actual1.csv', index=False)
#predict=pd.DataFrame(data=sol, columns=['predict'])
sol.to_csv('predict.csv',index=False)
sol1.to_csv('predict1.csv',index=False)





