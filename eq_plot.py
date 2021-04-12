import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("data.csv")
sns.scatterplot(data=df, x='PS',y='Re')
plt.show()
