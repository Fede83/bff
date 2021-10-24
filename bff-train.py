import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import sklearn
from sklearn.model_selection import train_test_split
#from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from keras_visualizer import visualizer
#import seaborn as sns

print(tf.__version__)

url = 'bff.csv'

usecols = ["negative_adjectives_component", "social_order_component", "action_component",
           "positive_adjectives_component", "joy_component", "affect_friends_and_family_component",
           "fear_and_digust_component", "politeness_component", "polarity_nouns_component", "polarity_verbs_component",
           "virtue_adverbs_component", "positive_nouns_component", "respect_component", "trust_verbs_component",
           "failure_component", "well_being_component", "economy_component", "certainty_component",
           "positive_verbs_component", "objects_component", "BFF_Label"]

raw_dataset = pd.read_csv(url,
  header = 0,
  usecols = usecols,
  na_values='?',
  comment='\t',
  sep=';',
  skipinitialspace=True)

df = raw_dataset.copy()
df.dropna(subset=usecols, inplace=True)

print("Sampling 0s")
n=120

print("Preparing the dataset. Elements per class:",n)
dataset = pd.concat( [
  df[df['BFF_Label']=="A"].sample(n=n,replace=True),
  df[df['BFF_Label']=="C"].sample(n=n,replace=True),
  df[df['BFF_Label']=="E"].sample(n=n,replace=True),
  df[df['BFF_Label']=="N"].sample(n=n,replace=True),
  df[df['BFF_Label']=="O"].sample(n=n,replace=True)
  ], ignore_index=True)

dataset["BFF_Label"].replace({"A": 0, "C": 1, "E": 2, "N": 3, "O": 4}, inplace=True)

print("Shuffling")
dataset = dataset.sample(frac=1).reset_index(drop=True)

y=dataset.pop("BFF_Label")
x=dataset

print("Preparing train set and test set")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#@title Define the plotting function
def plot_curve(epochs, hist, list_of_metrics):
 
# Plotting the Graph
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()
  plt.savefig("bff.png")


def create_model(my_learning_rate):

    model = tf.keras.models.Sequential()

    # The features are stored in a two-dimensional 28X28 array.
    # Flatten that two-dimensional array into a a one-dimensional
    # 784-element array.
    model.add(tf.keras.layers.InputLayer(input_shape=(20,)))

    # Define the first hidden layer.
    model.add(tf.keras.layers.Dense(units=48, activation='relu'))


    # Define a dropout regularization layer.
    model.add(tf.keras.layers.Dropout(rate=0.2))

    # Define the first hidden layer.
    #model.add(tf.keras.layers.Dense(units=24, activation='relu'))


    # Define a dropout regularization layer.
    #model.add(tf.keras.layers.Dropout(rate=0.2))

    # Define the output layer.
    model.add(tf.keras.layers.Dense(units=5, activation='softmax'))

    # Construct the layers into a model that TensorFlow can execute.
    # Notice that the loss function for multi-class classification
    # is different than the loss function for binary classification.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model

def train_model(model, x_train, y_train, epochs,
                batch_size=None, validation_split=0.1
                ):
    """Train the model by feeding it data."""

    
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, shuffle=True,
                        validation_split=validation_split,
                        callbacks=[es]
                        )


    # To track the progression of training, gather a snapshot
    # of the model's metrics at each epoch.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

# The following variables are the hyperparameters.
learning_rate = 0.001
epochs = 4096
batch_size = 16
validation_split = 0.1

# Establish the model's topography.
my_model = create_model(learning_rate)


# Train the model on the normalized training set.
epochs, hist = train_model(my_model, x_train, y_train,
                           epochs, batch_size, validation_split)

# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ['accuracy','loss','val_accuracy','val_loss']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Save the model for predicions
my_model.save('bff_model')

# Evaluate against the test set.
print("Evaluate on test data")
results = my_model.evaluate(x_test, y_test, batch_size=batch_size)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 5 samples")
predictions = my_model.predict(x_test[:5])
print("predictions shape:", predictions.shape)
print(y_test[:5])
print("------------------")
print(predictions)

print(my_model.summary())
visualizer(my_model, filename='bff-train_model', format='png', view=False)

#Generate confusion matrix
y_pred = my_model.predict(x_test)

print("y_test")
print(y_test.argmax(axis=1))
print("-------------")
print("y_pred")
print(y_pred.argmax(axis=1))

conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), normalize='true')

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=round(conf_matrix[i, j],2), va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.savefig("bff-confusion.png")

