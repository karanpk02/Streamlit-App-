import pandas as pd
df = pd.read_csv(r"/home/karan/Downloads/Iris.csv")
#import numpy as np
#import seaborn as sn
import pickle

b = []
for i in df.keys():
  b.append(i)
  
b.remove('Id')
b.remove('Species')

X = df[b].values#array of features
y = df['Species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=43)

from sklearn.preprocessing import StandardScaler ## standrard scalig 
scaler = StandardScaler() #initialise to a variable
scaler.fit(X_train,y_train) # we are finding the values of mean and sd from the td
X_train_scaled = scaler.transform(X_train) # fit (mean, sd) and then transform the training data
X_test_scaled = scaler.transform(X_test) # transform the test data

from sklearn.linear_model import LogisticRegression #main code that build the LR model 
logistic_regression = LogisticRegression() #initialise the required package
logistic_regression.fit(X_train_scaled,y_train) #magic happens - best values of betas - training/learning happens here
y_pred=logistic_regression.predict(X_test_scaled)

from sklearn.ensemble import RandomForestClassifier
# Instantiate model 
rf = RandomForestClassifier(n_estimators= 100, max_depth = 3)

# Train the model on training data
rf.fit(X_train_scaled, y_train);

# Use the forest's predict method on the test data
y_pred = rf.predict(X_test_scaled)

pickle.dump(logistic_regression, open('log_model.pkl','wb'))  
pickle.dump(rf, open('rf_model.pkl','wb'))