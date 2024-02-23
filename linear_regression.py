# #AI code to implement linear regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2)

# Train the model on the training set
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot predicted vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Predicted vs. Actual Values")

# Plot the linear regression line
plt.plot([0, 350], [0, 350], '--k')

# Print the results
print("Mean Squared Error:", mse)
print("Weight:", model.coef_)
print("Intercept:", model.intercept_)
print("Accuracy Score:", r2)

# Show the plot
plt.show()



































# import matplotlib as plt
# import numpy as np
# from sklearn  import datasets, linear_model
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
# Diabetes=datasets.load_diabetes()
# #print(Diabetes)
# # print(Diabetes)
# # print(Diabetes.target_filename)
# # print(Diabetes.data)
# # print(Diabetes.target)
# Diabetes_X_train=Diabetes.data[:-30]
# Diabetes_Y_train=Diabetes.target[:-30]
# Diabetes_X_test=Diabetes.data[-30:]
# Diabetes_Y_test=Diabetes.target[-30:]
# model=linear_model.LinearRegression()
# model.fit(Diabetes_X_train,Diabetes_Y_train)
# Diabetes_Y_Predict = model.predict(Diabetes_X_test)
# print("Mean Squared Error :",mean_squared_error(Diabetes_Y_test,Diabetes_Y_Predict))
# print("Weight ",model.coef_)
# print("Intercept ",model.intercept_)
# print("Acuuracy Score", r2_score(Diabetes_Y_test,Diabetes_Y_Predict))