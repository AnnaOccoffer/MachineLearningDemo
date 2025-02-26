# IMPORT SECTION
import numpy as np
import pandas as pd
# IMPORT SECTION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import seaborn as sns
from sklearn.datasets import load_iris
# END OF IMPORT SECTION

#MAIN
#importing the dataset: the iris dataset contains data of three species of flowers
dataset = load_iris() #lo recupera direttamente da internet, non serve cvs,...

#creating the dataframe
data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names) #le features (cioè le x)
data['target'] = dataset.target #il target (cioè le y)

# visualizing the first rows of the dataset
print(f"\n here are the first 5 rows of the dataset: \n{data.head()}")

X = data.iloc[:,:-1].values # all the columns except the last one
y = data["target"].values # the last column

# splitting the dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 101, stratify = y)
# note: the stratify parameter ensures that classes are well balanced between train and test

scaler = StandardScaler()
# we are going to scale ONLY the features (i.e. the X) and NOT the y!
X_train_scaled = scaler.fit_transform(X_train) # fitting to X_train and transforming them
X_test_scaled = scaler.transform(X_test) # transforming X_test. DO NOT FIT THEM!