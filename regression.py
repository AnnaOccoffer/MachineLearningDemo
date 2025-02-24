# --- IMPORT SECTION ---
import pandas as pd # for DataFrames
import numpy as np # for numpy array operations
import matplotlib.pyplot as plt # for visualization
import math
#import seaborn as sns

# importing all the needed functions from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
# --- END OF IMPORT SECTION ---

# --- MAIN CODE ---
#importing the dataset
path_to_data = "Salary_Data.csv"
data = pd.read_csv(path_to_data)
print(f"\n here are the first 5 rows of the dataset: \n{data.head()}")

X = data["YearsExperience"].values.reshape(-1, 1)
y = data["Salary"].values.reshape(-1, 1)

#using a plot to visualize the data
plt.title("Years of Experience vs Salary") # title of the plot
plt.xlabel("Years of Experience") # title of x axis
plt.ylabel("Salary") # title of y axis
plt.scatter(X, y, color = "red") # actual plot
#sns.regplot(data = data, x = "YearsExperience", y = "Salary")
plt.show() # renderize the plot to show it

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 101)

print(f"\nthe total X size is {X.shape[0]}")
print(f"\nthe X_train size is {X_train.shape[0]} and is the {X_train.shape[0] / X.shape[0]*100} % of the total X")
print(f"\nthe X_test size is {X_test.shape[0]} and is the {X_test.shape[0] / X.shape[0]*100} % of the total X")

print(f"\nthe total y size is {y.shape[0]}")
print(f"\nthe y_train size is {y_train.shape[0]} and is the {y_train.shape[0] / y.shape[0]*100} % of the total y")
print(f"\nthe y_test size is {y_test.shape[0]} and is the {y_test.shape[0] / y.shape[0]*100} % of the total y")

# Visualizing data before scaling
print(f"\n-- BEFORE SCALING -- X_train:\n{X_train[:5]}")
print(f"\n-- BEFORE SCALING -- y_train:\n{y_train[:5]}")
print(f"\n-- BEFORE SCALING -- X_test:\n{X_test[:5]}")
print(f"\n-- BEFORE SCALING -- y_test:\n{y_test[:5]}")
# Feature scaling
scaler = StandardScaler()
# we are going to scale ONLY the features (i.e. the X) and NOT the y!
X_train_scaled = scaler.fit_transform(X_train) # fitting to X_train and transforming them
X_test_scaled = scaler.transform(X_test) # transforming X_test. DO NOT FIT THEM!
# Visualizing data after scaling
print(f"\n-- AFTER SCALING -- X_train:\n{X_train_scaled[:5]}")
print(f"\n-- AFTER SCALING -- y_train:\n{y_train[:5]}")
print(f"\n-- AFTER SCALING -- X_test:\n{X_test_scaled[:5]}")
print(f"\n-- AFTER SCALING -- y_test:\n{y_test[:5]}")

# linear regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(f"\n after the training, the params for the regressor are: {model.coef_}")

plt.title("years of experience vs salary")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.scatter(X_test, y_test, color = "red", label = "real data")
plt.plot(X_test, y_pred, color = "blue", label = "predicted data")
plt.legend()
plt.show()

#evaluating the model
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n rmse is {rmse:.2f}")

# --- END OF MAIN CODE ---