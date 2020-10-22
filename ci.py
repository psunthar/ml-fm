import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



df=pd.read_csv('training1.csv')
Y=df['del_p'].values
X=df.drop(['del_p','R'], axis=1)


X=X.values

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=.1,random_state=101)
#re_ci=pd.DataFrame(data=X_train,columns=['velocity','dia1','density','viscoscity','dia2','Re','Ci'])
re=pd.DataFrame(data=X_test,columns=['velocity','dia1','density','viscoscity','dia2','Re'])

print(re)

X_test1=pd.read_csv('testing_fro_Re.csv')
X_test1.drop(['del_p','R'],axis=1,inplace=True)
#print(X_test1)
X_test1.to_csv('X_test1_for_ci_plot.csv',index=False)
re.to_csv('X_test_for_ci_plot.csv',index=False)


#print(re_ci)
