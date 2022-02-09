import pandas as pd
import numpy as np
import time
from numpy.ma import array

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 17)

# Preprocess
# Read Data
print("Reading....")
start = time.time()
df = pd.read_csv('outputs/data/highways/highway_data.csv')
df = df.sort_values(['Vehicle_ID'])
df = df.drop(columns=['Unnamed: 0'])
df = df.reset_index(drop=True)

print("Columns:", df.columns)
print(df)

# test_data.drop(test_data.index[test_data['vehicleID'] == 0], inplace=True)
# test_data.drop(test_data.index[test_data['vehicleID'] == 4], inplace=True)
# test_data.drop(test_data.index[test_data['vehicleID'] == 684], inplace=True)
# test_data.drop(test_data.index[test_data['vehicleID'] == 689], inplace=True)
# test_data.drop(test_data.index[test_data['vehicleID'] == 2048], inplace=True)
# test_data.drop(test_data.index[test_data['vehicleID'] == 2403], inplace=True)
# test_data.drop(test_data.index[test_data['vehicleID'] == 2456], inplace=True)
# test_data.drop(test_data.index[test_data['vehicleID'] == 2641], inplace=True)
# test_data.drop(test_data.index[test_data['vehicleID'] == 2864], inplace=True)


test_x = np.zeros((1, 10, 15))
test_y = np.zeros((1, 2))
temp_car = pd.DataFrame()
current_car = 1
counter = 0
countercounter = 0

total = 0
for i, new_df in df.groupby(level=0):
    # i is the iteration
    # new_df is the dataframe with the car i)

    # print(new_df)

    temp_numpy = new_df.to_numpy()
    if(current_car == temp_numpy[0][1]):   # if the same car
        counter += 1
    else:   # if next car
        total += counter
        if(counter < 257):
            print("# of data points: ", counter, "    for car: ", current_car)
            print("Dropping...")
            df.drop(df.index[df['Vehicle_ID'] == current_car], inplace=True)
            countercounter+=1
        counter = 0
        current_car = temp_numpy[0][1]     # update to new car

print("Number of cars that dont meet the criteria: ", countercounter)
print("AVERAGE: ", total/3366)

df.to_csv('highway_data_2.csv')