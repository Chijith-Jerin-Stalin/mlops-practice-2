import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv("Cancer_Data.csv")
X = df.iloc[:,2:]
le = LabelEncoder()
y = le.fit_transform(df["diagnosis"])

X_train, X_test, y_train,y_test = train_test_split(X,y,train_size=0.25,random_state=0)

randomf_model = RandomForestClassifier(n_estimators=200,random_state=42)
linearr_model = LinearRegression()
decision_model = DecisionTreeClassifier(criterion="entropy",random_state=0)

randomf_model.fit(X_train,y_train)

with open("randomf_model.pkl","wb") as f:
    pickle.dump(randomf_model,f)

with open("leencoder.pkl","wb") as f:
    pickle.dump(le,f)



print("Random Forest Model has been trained successfully")