# %%
# chatGPT tensorflow  - bibox Seite 251

# Q5 mit HeNormal Initialisierung

# %%
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# hh add from https://www.tensorflow.org/guide/keras/sequential_model
import keras
from keras import layers

# hh initializer
from keras.initializers import HeNormal

# %%


# %%
# geändert: 2 3 4 5 6
# Eingabedaten mit .constant
# In diesem Fall haben wir nur fünf Datenpunkte in einer Dimension und versuchen, eine Grenze zu lernen, die die Werte in zwei Klassen teilt.

# Neue Daten erstellen mit Rauschen
x_train = tf.constant([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=tf.float32)
y_train = tf.constant([[0], [0], [1], [1], [1]], dtype=tf.float32)

# Rauschen hinzufügen
noise = tf.random.normal(shape=x_train.shape, mean=0.0, stddev=0.2)
x_train_noisy = x_train + noise

# Q3: Originale Daten skalieren
x_train_scaled = (x_train - tf.reduce_min(x_train)) / (tf.reduce_max(x_train) - tf.reduce_min(x_train))

print(x_train)
print("--")
print(y_train)

# %%
# Modell erstellen
# Das Modell hat nur eine Schicht mit einer Neuron und Sigmoid-Aktivierung, da wir eine binäre Klassifikation haben.

# Q5 Für die Initialisierung können wir eine andere Methode verwenden, z. B. He-Initialisierung, die oft bei ReLU-Aktivierungsfunktionen hilfreich ist, da sie die Gewichte so verteilt, dass die Gradienten stabiler bleiben.

model = keras.Sequential()
model.add(keras.Input(shape=(1,)))
model.add(layers.Dense(units=8, activation='relu', kernel_initializer=HeNormal()))  # Zusätzliche Schicht mit 8 Neuronen
model.add(layers.Dense(units=1, activation='sigmoid'))

# model = Sequential([
    # OLD  Dense(units=1, activation='sigmoid', input_shape=(1,))
# ])


# %%
# Q4 Lernrate 0.1

# Kompilieren des Modells
# Hier spezifizieren wir den Optimierungsalgorithmus, den Verlust (Loss) und die Metrik zur Modellbewertung.
# Wir verwenden binary_crossentropy als Verlustfunktion, da es sich um eine binäre Klassifikationsaufgabe handelt, und sgd (Stochastic Gradient Descent) als Optimierer.


# Kompilieren mit einer kleineren Lernrate
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),  # Q4 Lernrate
              loss='binary_crossentropy',
              metrics=['accuracy'])

# OLD  model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])


# %%
# das Modell grafisch darstellen.
# Hinweis: Damit dieser Befehl funktioniert, muss Graphviz installiert sein.

plot_model(model, show_shapes=True, show_layer_names=True)


# %%
# Modell trainieren
# Hier trainieren wir das Modell für 100 Epochen. verbose=0 sorgt dafür, dass keine Ausgaben während des Trainings angezeigt werden.

# Q3 scaled:
history = model.fit(x_train_scaled, y_train, epochs=200)
# OLD  history = model.fit(x_train, y_train, epochs=100, verbose=0)


# %%
# eval
# Modell bewerten

# Q3 scaled
loss, accuracy = model.evaluate(x_train_scaled, y_train, verbose=0)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


# %%
# Vorhersagen für die Eingabedaten
# Das Modell gibt eine Wahrscheinlichkeit zwischen 0 und 1 aus (wegen der Sigmoid-Aktivierung). Diese Werte können als Wahrscheinlichkeiten für die Klasse "1" interpretiert werden.
print(x_train)
print("--")

predictions = model.predict(x_train)
print("Predictions:", predictions)


# %%
# Trainingsverlauf visualisieren
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Model Training Performance with Scaled Inputs')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()


# %%



