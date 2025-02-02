#import the necessary python libraries
import pandas as pd
import numpy as np
from sklearn import linear_model

#Brief of this program
print("This code is to predict value of independent variable Range based on dependent variables like Velocity, Time, Angel and Height \n")


#Get input values of dependenct variables from users
v_velocity=input("Enter Velocity: ")
v_time=input("Enter Time: ")
v_angle=input("Enter Angle: ")
v_height=input("Enter Height: ")

#Below path is of data file where values of Velocity, time, angle, height and range are available. These values are used to train and test the linear regression model

url = "https://raw.githubusercontent.com/trul2010/linear_regression/main/projectile_motion.csv"

raw_ds=pd.read_csv(url)

projectile_ds=raw_ds.iloc[:, : 5]

# Dependent variable data frame
range_ds=projectile_ds['Range']

projectile_ds.head(5)

independent_ds=projectile_ds[['Velocity','Time','Angle','Height']]

# Independent variable data frame
independent_ds.head(5)


#Values are fed to Linear regression model
regr = linear_model.LinearRegression()
regr.fit(independent_ds, range_ds)

#Build the dataframe for which range needs to be predicted. These values are provided by the user
data = {
  "Velocity":v_velocity,
  "Time":v_time,
  "Angle":v_angle,
  "Height":v_height

}

#load data into a DataFrame object:
df = pd.DataFrame(data,index=[0])

#Predit the range
predict_range=regr.predict(df)

print("\n The predicted value of range for provided dependent variable values is :",predict_range)