# %%
# Test von graphviz bzw. dtreeviz mit Daten penguins

# %%
import numpy as np
import pandas as pd
import seaborn as sns

import dtreeviz


# %%

from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# %%
# Daten laden, hier aus seaborn

df = sns.load_dataset("penguins")

# %%
# Anzeige (beachte ohne print: jupyter zeigt den Wert der letzten Zeile einer Zelle)
df

# %%
# Lösche nicht gebrauchte Spalten island, sex
df = df.drop(columns=['island', 'sex'])
# df = df.drop(['island', 'sex'], axis=1)     # weniger gut lesbar

# %%
# Nach dem Entfernen der Spalten gibt es noch zwei Zeilen mit fehlenden Werten
df.isna().sum()

# %%
# Entfernen von Zeilen mit fehlenden Werten
df = df.dropna()

# %%
# nochmal auf fehlende Werte prüfen
df.isna().sum()

# %%
# Unterteilen in Trainings- und Test-Daten (inkl. Unterteilung in X und y)

# %%
# Anmerkung zum Video bei 5:51 "Datensatz" sollte eher "dataset" bzw. Stichprobe sein
from sklearn.model_selection import train_test_split

# %%
# Unterteilung in Features X
X = df.drop(columns=['species'])
# X = df.drop(['species'], axis=1)    # schlechter lesbar

# %%
# Unterteilung in Labels y
y_df = df['species']

# %%
# LabelEncoding für die Visualisierung
le = LabelEncoder()
le.fit(list(y_df.unique()))
y = le.transform(y_df)

# list of classes: le.classes_

# %%
# Unterteilen in Trainings- und Test-Daten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,   # 20% Testdaten
                                                    random_state=119)

# %%
# Anzeige
X_train

# Ergebnis: 266 rows × 4 columns

# %%
# Anzeige
X_test

# Ergebnis: 67 rows × 4 columns

# %%
# Anzeige
y_test

# Ergebnis: 67, Datentyp ist Series anstatt DataFrame, Anmerkung: print(type(y_test))

# %%
# init decision tree classifier
clf = tree.DecisionTreeClassifier(max_depth=2, random_state=42)

# %%
# trainiere den classifier auf den Traningsdaten
clf.fit(X_train.values, y_train)

# %%
# Vorhersagen machen für die Testdaten, nur aus Merkmalen X_test (ohne y_test zu verwenden)
predictions = clf.predict(X_test.values)

# %%
# Anzeige
predictions

# Ergebnis: Es werden species ausgegeben, ohne dass wir Labels y_test benutzt haben

# %%
# Wie gut clf gerechnet hat: Vergleich von predictions zu den wahren Labels y_test
y_test


# %%
# Genauigkeit bestimmen
# teile Anzahl richtig erkannter Pinguine durch alle Pinguine (per sklearn)
# Berechnung der Genauigkeit "accuracy" (zur Theorie suche "Verwechslungsmatrix"/confusion matrix/evaluation)
accuracy_score(predictions, y_test)

# Ergebnis decision tree mit 97% ist besser als KNN mit 88%

# %% [markdown]
# ## Grafik

# %%
# Grafik eines decision tree
viz = dtreeviz.model(clf,
                X_train=X_train,
                y_train=y_train,
                target_name='species',
                feature_names=X_train.columns,     # iris.feature_names,
                class_names=list(le.classes_)
                # title="Decision Tree - Iris data set"
                )
viz.view()

# see button expand plot -> zoom etc.


