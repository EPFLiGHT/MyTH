import os
import shutil
import random

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

seed = 12
np.random.seed(seed)
random.seed(seed)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 1. Read the data
data = pd.read_csv("train.csv")

# 2. Extract images in a cardiomegaly (positive) class
data_cardio = []

for index, row in data.iterrows():
    if row['Cardiomegaly'] == 1.0:
      data_cardio.append(row['Path'])

print('Number of images in a positive class:', len(data_cardio))

# 3. Extract images not in cardiomegaly (negative) class
data_not_cardio = []

for index, row in data.iterrows():
    if row['Path'] not in data_cardio:
      data_not_cardio.append(row['Path'])

imbalance = len(data_not_cardio)/len(data_cardio)

print('Number of images in a negative class:', len(data_not_cardio), '\nImbalance:', imbalance)

# 4. Split data
random.shuffle(data_not_cardio)
num_train = 24300 # a number of images in each class in the training set (balanced) with 0.9/0.1 train/test split
num_test = (len(data_cardio) - num_train) * imbalance # number of images in the negative class in the test set (imbalanced)
data_not_cardio_train = data_not_cardio[:num_train]
data_not_cardio_test = data_not_cardio[num_train:int(num_train+num_test)]

train_data, test_data = train_test_split(data_cardio, test_size=0.1, random_state=12) # split data in a positive class

# 5. Save data
os.makedirs('cardiomegaly/test/positive')
os.makedirs('cardiomegaly/test/negative')
os.makedirs('cardiomegaly/train/positive')
os.makedirs('cardiomegaly/train/negative')
os.makedirs('cardiomegaly/push/positive')
os.makedirs('cardiomegaly/push/negative')

# save train/positive data
for num, row in enumerate(train_data):
    shutil.copy(row, f'cardiomegaly/train/positive/{num}.jpg')

# save train/negative
for num, row in enumerate(data_not_cardio_train):
    shutil.copy(row, f'cardiomegaly/train/negative/{num}.jpg')

# save test/positive
for num, row in enumerate(test_data):
    shutil.copy(row, f'cardiomegaly/test/positive/{num}.jpg')

# save test/negative
for num, row in enumerate(data_not_cardio_test):
    shutil.copy(row, f'cardiomegaly/test/negative/{num}.jpg')

# save push data, a subset of training set used for ptototype visualization
push_inds_p = np.random.permutation(len(train_data))[:3000]
n=0
for num, row in enumerate(train_data):
  if num in push_inds_p:
      shutil.copy(row, f'cardiomegaly/push/positive/{n}.jpg')
      n+=1

push_inds_n = np.random.permutation(len(data_not_cardio_train))[:3000]
n=0
for num, row in enumerate(data_not_cardio_train):
  if num in push_inds_n:
      shutil.copy(row, f'cardiomegaly/push/negative/{n}.jpg')
      n+=1

# 6. Check class balance
print('Ratio of the number of images in the negative class to the number of images in the positive class in:', 
      '\ntraining set:', len(os.listdir('cardiomegaly/train/negative'))/len(os.listdir('cardiomegaly/train/positive')), 
      '\ntest set:', len(os.listdir('cardiomegaly/test/negative'))/len(os.listdir('cardiomegaly/test/positive')))

