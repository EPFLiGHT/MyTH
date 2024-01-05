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

# 2. Extract images in a pleural effusion (positive) class
data_effusion = []

for index, row in data.iterrows():
    if row['Pleural Effusion'] == 1.0:
      data_effusion.append(row['Path'])

print('Number of images in a positive class:', len(data_effusion))

# 3. Extract images not in pleural effusion (negative) class
data_not_effusion = []

for index, row in data.iterrows():
    if row['Path'] not in data_effusion:
      data_not_effusion.append(row['Path'])

imbalance = len(data_not_effusion)/len(data_effusion)

print('Number of images in a negative class:', len(data_not_effusion), '\nImbalance:', imbalance)

# 4. Split data
# since the size of positive class is large, we take only 20616 images
random.shuffle(data_effusion)
data_effusion = data_effusion[:20616]

random.shuffle(data_not_effusion)
num_train = 18554 # a number of images in each class in the training set (balanced) with 0.9/0.1 train/test split
num_test = (len(data_effusion) - num_train) * imbalance # number od images in the negative class in the test set (imbalanced)
data_not_effusion_train = data_not_effusion[:num_train]
data_not_effusion_test = data_not_effusion[num_train:int(num_train+num_test)]

train_data, test_data = train_test_split(data_effusion, test_size=0.1, random_state=12) # split data in a positive class

# 5. Save data
os.makedirs('effusion/test/positive')
os.makedirs('effusion/test/negative')
os.makedirs('effusion/train/positive')
os.makedirs('effusion/train/negative')
os.makedirs('effusion/push/positive')
os.makedirs('effusion/push/negative')

# save train/positive data
for num, row in enumerate(train_data):
    shutil.copy(row, f'effusion/train/positive/{num}.jpg')

# save train/negative
for num, row in enumerate(data_not_effusion_train):
    shutil.copy(row, f'effusion/train/negative/{num}.jpg')

# save test/positive
for num, row in enumerate(test_data):
    shutil.copy(row, f'effusion/test/positive/{num}.jpg')

# save test/negative
for num, row in enumerate(data_not_effusion_test):
    shutil.copy(row, f'effusion/test/negative/{num}.jpg')

# save push data, a subset of training set used for ptototype visualization
push_inds_p = np.random.permutation(len(train_data))[:3000]
n=0
for num, row in enumerate(train_data):
  if num in push_inds_p:
      shutil.copy(row, f'effusion/push/positive/{n}.jpg')
      n+=1

push_inds_n = np.random.permutation(len(data_not_effusion_train))[:3000]
n=0
for num, row in enumerate(data_not_effusion_train):
  if num in push_inds_n:
      shutil.copy(row, f'effusion/push/negative/{n}.jpg')
      n+=1

# 6. Check class balance
print('Ratio of the number of images in the negative class to the number of images in the positive class in:', 
      '\ntraining set:', len(os.listdir('effusion/train/negative'))/len(os.listdir('effusion/train/positive')), 
      '\ntest set:', len(os.listdir('effusion/test/negative'))/len(os.listdir('effusion/test/positive')))

