import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pk

df=pd.read_csv('training1.csv')
#print(df)
df.drop(['R','Re','del_p','dia2','velocity','viscosity'],axis=1,inplace=True)
from pylab import savefig
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

#df1=pd.read_csv('/home/ram/Desktop/R=2.6/training1.csv')
#Y=df1['del_p'].values
#X=df1.drop(['del_p','R','Re'], axis=1)

print(df)
X_train=np.array(df)

#X_train=pd.DataFrame(data=X_train, columns=['velocity','dia1','density','viscoscity','dia2'])
#R_V=np.random.uniform(1,100,9326)
#df['Random']=R_V

#sns.heatmap(df.corr(),annot=True)
#plt.show()
#scaled_vs_pca=pd.concat([X_train_scaled,X_train_scaled_pca],axis=1)


#print(df.head())
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_train_scaled=pd.DataFrame(data=X_train, columns=['density_s','dia1_s'])
#X_train_scaled.to_csv('X_train_scaled.csv')
#x=np.linspace(1,9326,9326)


#unscaled_vs_scaled=pd.concat([X_train_scaled,df],axis=1)

#sns.scatterplot(x='viscosity',y='viscoscity_s',data=unscaled_vs_scaled,color='r',label='Scaled_vs_Unscaled')


#plt.show()




###################################################3PCA_PLOTS##################


from sklearn.decomposition import PCA

pca_train=PCA()

pca_train.fit(X_train)
X_train_pca=pca_train.transform(X_train)

pk.dump(pca_train, open("pca.pkl","wb"))


X_train_scaled_pca=pd.DataFrame(data=X_train_pca, columns=['p1','p2'])
#sns.heatmap(X_train_scaled.corr(),annot=True)
#plt.show()
#scaled_vs_pca=pd.concat([X_train_scaled,X_train_scaled_pca],axis=1)
bgc=pd.concat([X_train_scaled,X_train_scaled_pca], axis=1)

sns.heatmap(bgc.corr(),annot=True)
plt.show()


#x=np.linspace(1,9326,9326)
#sns.scatterplot(scaled_vs_pca['velocity_s'],scaled_vs_pca['p5'],data=scaled_vs_pca)

#sns.scatterplot(x,scaled_vs_pca['viscoscity_s'],label='scaled')

#plt.show()

