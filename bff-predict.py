
# https://www.tensorflow.org/tutorials/keras/regression

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

url = "reviews.csv"

usecols = ["filename","negative_adjectives_component", "social_order_component", "action_component",
           "positive_adjectives_component", "joy_component", "affect_friends_and_family_component",
           "fear_and_digust_component", "politeness_component", "polarity_nouns_component", "polarity_verbs_component",
           "virtue_adverbs_component", "positive_nouns_component", "respect_component", "trust_verbs_component",
           "failure_component", "well_being_component", "economy_component", "certainty_component",
           "positive_verbs_component", "objects_component"]

raw_dataset = pd.read_csv(url,
  header = 0,
  usecols = usecols,
  na_values='?',
  comment='\t',
  sep=';',
  skipinitialspace=True)

df = raw_dataset.copy()
df.dropna(subset=usecols, inplace=True)
filenames=df.pop("filename")
x_predict = df.copy()


my_model = tf.keras.models.load_model('bff_model')

y_predict = my_model.predict(x_predict)

bffs=["A","C","E","N","O"]
print("filename;rating;bff;max;A;C;E;N;O")
for idx in range(len(filenames)):
  rating = int(filenames[idx].split("_")[1].replace(".txt",""))


  max_value = np.amax(y_predict[idx])
  max_index = np.where(y_predict[idx] == max_value)[0][0]
  bff=bffs[max_index]

  print(filenames[idx]+";"+str(rating)+";"+bff+";"+str(max_value)+";"+";".join(map(str,y_predict[idx])))
