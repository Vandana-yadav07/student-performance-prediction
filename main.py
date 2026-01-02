import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("student_data.csv")
x=data[["study_hours","attendance"]]
y=data["marks"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state= 42)

model = LinearRegression()
model.fit(x_train, y_train)

prediction = model.predict([[6,75]])

print("Predicted marks for study hours= 6 and attendance = 75%:")
print(prediction[0])