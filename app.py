from flask import Flask
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("Cancer_Data.csv")
X = df.iloc[:,2:]
le = LabelEncoder()
y = le.fit_transform(df["diagnosis"])

X_train, X_test, y_train,y_test = train_test_split(X,y,train_size=0.25,random_state=0)



app = Flask(__name__)

with open("randomf_model.pkl","rb") as f:
    model = pickle.load(f)

with open("leencoder.pkl","rb") as f:
    le = pickle.load(f)

@app.route("/")
def home():
    return "RandomForest Model is Running"

@app.route("/predict",methods=["GET"])
def predict():
    pred = model.predict([[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]])
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    cancer = le.inverse_transform(pred)
    return  f"The Cancer Cell Prediction: {cancer}\n and the Accuracy is:  {accuracy}"


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)