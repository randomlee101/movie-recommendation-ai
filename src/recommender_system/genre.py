import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential


def find_movie_genre(data_frame, movieId):
    return data_frame[
           data_frame.index[data_frame["movieId"] == movieId][0] - 1:data_frame.index[data_frame["movieId"] == movieId][
               0]]["genres"]


df = pd.read_csv("movies.csv")
df_ratings = pd.read_csv("ratings.csv")
mol_prime = []
mol = []

found_genres = []

for x in range(df.__len__()):
    splits = df["genres"][x].split("|")
    mol_prime.append(splits)
    mol.extend(splits)

unique_mol = np.unique(mol)


def true_or_false(value="", new_array=None):
    if new_array is None:
        new_array = []
    return 1 if new_array.__contains__(value) else 0


def compile_single_list(mol_array=None):
    if mol_array is None:
        mol_array = []
    return list(map(lambda m: true_or_false(value=m, new_array=mol_array), unique_mol))


df["genres"] = (list(map(compile_single_list, mol_prime)))

for y in range(1, df_ratings.__len__() + 1):
    movie_id = df_ratings[y - 1:y]["movieId"].item()
    result = list(find_movie_genre(data_frame=df, movieId=movie_id))
    if result.__len__() < 1:
        found_genres.append(result)
    else:
        found_genres.extend(result)

df_ratings["genres"] = found_genres
df_ratings = df_ratings[df_ratings['genres'].apply(lambda e: len(e) > 0)]

x2 = df_ratings[["userId", "movieId", "genres"]]
y = df_ratings[["rating"]]


def padding(item):
    return item if not (item.shape[0] == 0) else np.pad(item, (0, 20), 'constant')


# Extract the integer and nested list from each sublist
integers = np.array([item[:2] for item in np.asarray(x2)])

# Convert the nested lists to NumPy arrays and stack them horizontally
arrays = [padding(np.asarray(item[2])) for item in np.asarray(x2)]
stacked = np.vstack(arrays)

# Concatenate the integers and the nested lists horizontally
x = np.hstack([integers, stacked])
y = np.array(y)

x = tf.constant(x, dtype=tf.float32)
y = tf.constant(y, dtype=tf.float32)

x_train, x_val = tf.split(x, [80497, 20124])
y_train, y_val = tf.split(y, [80497, 20124])

model = Sequential(
    [
        tf.keras.layers.BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')
    ]
)

model.compile(optimizer='adam', metrics=['mean_absolute_error', 'accuracy'], loss='mean_absolute_error')
model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_val, y_val))
model.summary()

# use the trained model to predict ratings on the test dataset
predictions = model.predict(x_val)
print(predictions)
print(y_val)
