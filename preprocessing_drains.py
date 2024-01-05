import os
import shutil
import random
import glob

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from numpy.random.mtrand import randint
import Augmentor

seed = 12
np.random.seed(seed)
random.seed(seed)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
### Due to the small size of the drains dataset, we augment it using rotation, flip, skew, and shear operations ###

# 1. Read the data
data = pd.read_csv("train.csv")
drains = pd.read_csv('drains_376.csv')

# 2. Augment the data
class_name = 'Drain'
for index, row in drains.iterrows():
    fd = os.path.dirname(row['Path']) + '/'
    if row[f'{class_name}'] == 1.0:
      # rotation
      try:
        p = Augmentor.Pipeline(source_directory=fd)
        p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
        p.flip_left_right(probability=0.5)
        for i in range(2):
            p.process()
        del p
        # skew
        p = Augmentor.Pipeline(source_directory=fd)
        p.skew(probability=1, magnitude=0.3)  # max 45 degrees
        p.flip_left_right(probability=0.5)
        for i in range(2):
            p.process()
        del p
        # shear
        p = Augmentor.Pipeline(source_directory=fd)
        p.shear(probability=1, max_shear_left=20, max_shear_right=20)
        p.flip_left_right(probability=0.5)
        for i in range(2):
            p.process()
        del p
      except OSError:
        continue
      
# 3. Split data
train_data, test_data = train_test_split(drains, test_size=0.1, random_state=12) # split data in a positive class

# 4. Save data
os.makedirs('drains/test/positive')
os.makedirs('drains/test/negative')
os.makedirs('drains/train/positive')
os.makedirs('drains/train/negative')
os.makedirs('drains/push/positive')
os.makedirs('drains/push/negative')

# save push/positive and train/positive
n=0
for num, row in train_data.iterrows():
    shutil.copy(row.Path, f'drains/train/positive/{n}.jpg')
    shutil.copy(row.Path, f'drains/push/positive/{n}.jpg')
    n+=1

## add augmented data
for num, row in drains.iterrows():
  while len(os.listdir('drains/train/positive')) <= 4640: # a max number of images in a positive class in a training set (computed as approx. 18554/4 (number of clients))
    for name in glob.glob(os.path.dirname(row.Path) + '/output/*.jpg'):
      shutil.copy(name, f'drains/train/positive/{n}.jpg')
      n+=1

# save test/positive
n=0
for num, row in test_data.iterrows():
    shutil.copy(row.Path, f'drains/test/positive/{n}.jpg')
    n+=1

# save train/negative
negative_data = []
for i in range(4700): # 4640 images for training set + 60 images for test set (to keep imbalance in the test set 1.6)
  sample = randint(0, len(data))
  if data['Path'][sample] not in drains['Path']:
    negative_data.append(data['Path'][sample])

random.shuffle(negative_data)
for num, row in enumerate(negative_data[0:4640]):
  shutil.copy(row, f'drains/train/negative/{num}.jpg')

# save test/negative
for num, row in enumerate(negative_data[4640:]):
  shutil.copy(row, f'drains/test/negative/{num}.jpg')

# save push/negative
random.shuffle(negative_data[:4640])
for num, row in enumerate(negative_data[0:338]):
  shutil.copy(row, f'drains/push/negative/{num}.jpg')