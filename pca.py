import pandas as pd
import numpy as np
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
X_testa=pd.DataFrame(data=X_test, columns=['velocity','dia1','density','viscoscity','dia2'])
#X_testa=pd.DataFrame(data=X_test, columns=['velocity','dia1','density','viscoscity','dia2'])


pca_train=PCA()

pca_train.fit(X_train)
X_train_pca=pca_train.transform(X_train)# transforming training data to PCA components
#print(X_train_pca)
print(pca_train.explained_variance_ratio_)# explain variance in a feature

X_test1=pd.read_csv('/home/ram/Desktop/R=2.6/testing_fro_Re.csv')
Actual1=pd.DataFrame(data=X_test1,columns=['del_p'])
X_test1.drop(['del_p','Re','R'],axis=1,inplace=True)
X_test1=scaler.transform(X_test1.values)
Actual1.to_csv('actual1.csv', index=False)
X_test1=pd.DataFrame(data=X_test1,columns=['velocity','dia1','density','viscoscity','dia2'] )

X_test1.to_csv('X_test1.csv', index=False)


pca_test=PCA()
XTEST=pd.concat([X_testa,X_test1],axis=0)
pca_test.fit(XTEST)                                         #fiting whole  testing data for PCA
X_test_pca=pca_test.transform(X_testa)                      #transforming  testing data_inside training range
X_test1_pca=pca_test.transform(X_test1)                     #transforming testing data out of training range_Re
print(X_test1_pca)

X_testa.to_csv('testing_X.csv',index=False)

Y_testa=pd.DataFrame(data=y_test, columns=['del_p'])
Y_testa.to_csv('testing_Y.csv',index=False)
