# %%
# penguins aus Youtube Fabian Rappert
# https://www.youtube.com/watch?v=e6vPt_e9sRw

# Beachte Minuten-Angaben als Verweis auf das Video, z.B. "Bei 6:03 ..."

# %%
import numpy as np
import pandas as pd
import seaborn as sns


# %%
# Daten laden, hier aus seaborn

df = sns.load_dataset("penguins")

# %%
# Anzeige (beachte ohne print: jupyter zeigt den Wert der letzten Zeile einer Zelle)
df

# %%
# Visualisierung  (seaborn erweitert matplotlib für Datenanalyse)
# - Diagonale: Verteilung eines Merkmals
# - nicht-Diagonale: scatter plot aus zwei Merkmalen

sns.pairplot(df, hue = "species")

# %%
# Kommentar: erwarte Herausforderung bei der Unterscheidung von Adelie und Chinstrap

# %%
# prüfe fehlende Werte in irgendeiner Spalte, 1. Teil
df.isnull()

# %%
# zähle fehlende Werte, 2. Teil
df.isnull().sum()

# %%
# Kommentar: Bei so wenig fehlenden Werten können wir alle betroffenen Zeilen löschen
# Anmerkung hh: Besser würde erst die Spalte "sex" gelöscht (siehe 3 Zellen weiter),
# dann wären es nur noch 2 Zeilen mit fehlenden Werten

# %%
# Löschen der Zeilen inkl. Überschreiben desselben df
df = df.dropna()

# %%
# prüfe nochmal
df.isnull().sum()

# %%
# Behandlung von Text-Spalten (ausser species): island, sex
# Hier werden sie einfach gelöscht (alternativ könnte man sie in Zahlen wandeln: np.where, pd.get_dummies)
df = df.drop(columns=['island', 'sex'])
# df = df.drop(['island', 'sex'], axis=1)     # weniger gut lesbar

# %%
# Unterteilen in Trainings- und Test-Daten (inkl. Unterteilung in X und y)

# %%
# Anmerkung zum Video bei 5:51 "Datensatz" sollte eher "dataset" bzw. Stichprobe sein
from sklearn.model_selection import train_test_split

# %%
# bei 5:54 Erläuterung zu Features und Labels

# %%
# Unterteilung in Features X
X = df.drop(columns=['species'])
# X = df.drop(['species'], axis=1)    # schlechter lesbar

# %%
# Unterteilung in Labels y
y = df['species']

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
# import for K-nearest neighbor (KNN) classifier
from sklearn.neighbors import KNeighborsClassifier

# %%
# Bei 6:03 Erläuterung im Koordinatensystem mit zwei Merkmalen Schnabellänge und Gewicht
# Drei oder fünf nächste Nachbarn zu einem unbekannten Pinguin

# %%
# initialisiere einen Classifier clf  (Anmerkung, neben "clf" wäre auch "model" üblich)
clf = KNeighborsClassifier(n_neighbors=3)

# %%
# trainiere den KNN classifier auf den Traningsdaten
clf.fit(X_train, y_train)

# %%
# Vorhersagen machen für die Testdaten, nur aus Merkmalen X_test (ohne y_test zu verwenden)
predictions = clf.predict(X_test)

# %%
# Anzeige
predictions

# Ergebnis: Es werden species ausgegeben, ohne dass wir Labels y_test benutzt haben

# %%
# Wie gut clf gerechnet hat: Vergleich von predictions zu den wahren Labels y_test
# Bei 6:10 manueller Vergleich, schaue nur die ersten der Reihe nach an
y_test

# Ergebnis Beispiele: richtig bei 1. und 3., jedoch falsch bei 2. ("nicht schlimm, ganz normal")

# %%
# Genauigkeit bestimmen (letzter Schritt im Video)
# teile Anzahl richtig erkannter Pinguine durch alle Pinguine (per sklearn)
from sklearn.metrics import accuracy_score

# %%
# Berechnung der Genauigkeit "accuracy" (zur Theorie suche "Verwechslungsmatrix"/confusion matrix/evaluation)
accuracy_score(predictions, y_test)

# Ergebnis 88% ist viel besser als Zufall 33%

# %%
# Ende


