# Codsoft
# titanic survival prediction:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("tested.csv")
df.head()
df.head(10)
df.shape
df.describe()
df['Survived'].value_counts()
sns.countplot(x=df['Survived'], hue=df['Pclass'])
sns.countplot(x=df['Sex'], hue=df['Survived'])
df['Sex'].unique()
X= df[['Pclass', 'Sex']]
Y=df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression

log = LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)
pred = print(log.predict(X_test))
import warnings
warnings.filterwarnings("ignore")

res= log.predict([[3,2]])

if(res==0):
  print("So Sorry! Not Survived")
else:
  print("Survived")
# iris flower classificaation:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('IRIS.csv')
df.head()
df.describe()
df.info()
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']
for i in range(3):
x = df[df['species'] == species[i]]
plt.scatter(x['sepal_length'], x['sepal_width'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
for i in range(3):
x = df[df['species'] == species[i]]
plt.scatter(x['petal_length'], x['petal_width'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
for i in range(3):
x = df[df['species'] == species[i]]
plt.scatter(x['sepal_length'], x['petal_length'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()
df.corr()
corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.model_selection import train_test_split
X = df.drop(columns=['species'])
Y = df['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
print("Accuracy: ",model.score(x_test, y_test) * 100)
Accuracy: 100.0
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

# sales predictioon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("advertising.csv")
df.head()
df.head(10)
df.shape
df.describe()
sns.pairplot(df, x_vars=['TV', 'Radio','Newspaper'], y_vars='Sales', kind='scatter')
plt.show()
df['TV'].plot.hist(bins=10)
df['TV'].plot.hist(bins=10)
sns.heatmap(df.corr(),annot = True)
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size = 0.3,random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
res= model.predict(X_test)
print(res)
