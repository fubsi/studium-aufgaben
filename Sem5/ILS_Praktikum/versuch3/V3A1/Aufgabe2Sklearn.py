import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import time

# Pfad zu den CSV-Dateien
pfad_training = 'C:\\Users\\fbrze\\Documents\\EigeneDateien\\Programmierung_etc\\Git\\studium-aufgaben\\Sem5\\ILS_Praktikum\\versuch3\\V3A1\\training.csv'
pfad_test = 'C:\\Users\\fbrze\\Documents\\EigeneDateien\\Programmierung_etc\\Git\\studium-aufgaben\\Sem5\\ILS_Praktikum\\versuch3\\V3A1\\testing.csv'

# Einlesen der Daten
train_data = pd.read_csv(pfad_training)
test_data = pd.read_csv(pfad_test)

# Annahme: Erste Spalte ist das Label
X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]

X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# Konvertieren aller Werte in numerische Form (falls möglich) und Entfernen von NaNs
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Fehlende Werte (NaNs) durch Spaltenmittelwert ersetzen
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Standardisieren der Merkmale
skalierer = StandardScaler()
X_train = skalierer.fit_transform(X_train)
X_test = skalierer.transform(X_test)

# Labels in kategorische Form konvertieren
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# MLP-Klassifikator einrichten
mlp_pipeline = Pipeline([
    ('skalierer', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, solver='adam'))
])

# Kreuzvalidierung des MLP-Modells mit S = 3
cv = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)

# Messung der Trainingszeit
start_time = time.time()
mlp_bewertungen = cross_val_score(mlp_pipeline, X_train, y_train, cv=cv, scoring='accuracy')
mlp_trainingszeit = time.time() - start_time

# Training und Bewertung des MLP-Modells
mlp_pipeline.fit(X_train, y_train)
mlp_genauigkeit_test = mlp_pipeline.score(X_test, y_test)

print("MLP Accuracy (Cross Validation): ", mlp_bewertungen.mean())
print("MLP Accuracy (Testdata): ", mlp_genauigkeit_test)
print("MLP Training Time: {:.2f}s".format(mlp_trainingszeit))

# KNN-Klassifikator einrichten
knn_pipeline = Pipeline([
    ('skalierer', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# Messung der Trainingszeit
start_time = time.time()
knn_bewertungen = cross_val_score(knn_pipeline, X_train, y_train, cv=cv, scoring='accuracy')
knn_trainingszeit = time.time() - start_time

# Training und Bewertung des KNN-Modells
knn_pipeline.fit(X_train, y_train)
knn_genauigkeit_test = knn_pipeline.score(X_test, y_test)

print("KNN Accuracy (Cross Validation): ", knn_bewertungen.mean())
print("KNN Accuracy (Testdata): ", knn_genauigkeit_test)
print("KNN Training Time: {:.2f}s".format(knn_trainingszeit))
