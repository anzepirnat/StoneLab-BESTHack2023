import pickle
import pandas as pd
import random
import datetime

path = './data/food/GHG-emissions-by-life-cycle-stage-OurWorldinData-upload.csv'  
df = pd.read_csv(path)

food = [
    [],
    []
]

for i in range(len(df)):
    vrstica = df.iloc[i]
    food[0].append(vrstica['Food product'])
    suma = vrstica['Land use change'] + vrstica['Animal Feed'] + vrstica['Farm'] + vrstica['Processing'] + vrstica['Transport'] + vrstica['Packging'] + vrstica['Retail']
    food[1].append(suma)

food[0].append('Cappuccino')
food[1].append(9.65)
food[0].append('Americano')
food[1].append(16.5)

final_food = [

]


print([food[0][random.randint(0, len(food[0])-1)], food[1][random.randint(0, len(food[0])-1)], datetime.time(9, 56), datetime.date(2015, 7, 12)])

for i in range(4):
    final_food.append([food[0][random.randint(0, len(food[0])-1)], food[1][random.randint(0, len(food[0])-1)], datetime.time(9, 56), datetime.date(2015, 7, 13)])
for i in range(2):
    final_food.append([food[0][random.randint(0, len(food[0])-1)], food[1][random.randint(0, len(food[0])-1)], datetime.time(13, 34), datetime.date(2015, 7, 14)])
for i in range(12):
    final_food.append([food[0][random.randint(0, len(food[0])-1)], food[1][random.randint(0, len(food[0])-1)], datetime.time(6, 12), datetime.date(2015, 7, 15)])
for i in range(7):
    final_food.append([food[0][random.randint(0, len(food[0])-1)], food[1][random.randint(0, len(food[0])-1)], datetime.time(20, 45), datetime.date(2015, 7, 16)])

print(final_food)

pickle.dump(final_food, open('./data/food/final_food.pkl', 'wb'))