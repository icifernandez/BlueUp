from sklearn.ensemble import RandomForestClassifier
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Rutas de las carpetas donde están las imágenes
positive_folder = "C:/Users/icife/OneDrive/Escritorio/BlueUp/BlueUp-dron/archive/Positive"
negative_folder = "C:/Users/icife/OneDrive/Escritorio/BlueUp/BlueUp-dron/archive/Negative"

# Función para cargar las imágenes de una carpeta
def load_images_from_folder(folder, label, size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Cargar en escala de grises
        if img is not None:
            img_resized = cv2.resize(img, size)  # Redimensionar a 64x64 píxeles
            images.append(img_resized)
            labels.append(label)  # Etiqueta (1 = grietas, 0 = sin grietas)
    return images, labels

# Cargar imágenes de grietas (Positive) y sin grietas (Negative)
positive_images, positive_labels = load_images_from_folder(positive_folder, label=1)
negative_images, negative_labels = load_images_from_folder(negative_folder, label=0)

# Combinar ambos conjuntos de imágenes y etiquetas
X = np.array(positive_images + negative_images)  # Concatenar imágenes
Y = np.array(positive_labels + negative_labels)  # Concatenar etiquetas

# Aplanar las imágenes para pasarlas al modelo SVM (de 64x64 a 4096 valores)
X_flat = X.reshape(len(X), -1)

# Dividir en conjuntos de entrenamiento y prueba
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_flat, Y, test_size=0.2, random_state=42)


# Crear el modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Ajustar el modelo con el conjunto de entrenamiento
rf_model.fit(Xtrain, Ytrain)

# Realizar predicciones en el conjunto de prueba
yfit_rf = rf_model.predict(Xtest)

# Evaluar el rendimiento
print("Accuracy (Random Forest):", accuracy_score(Ytest, yfit_rf))
print("Classification Report (Random Forest):\n", classification_report(Ytest, yfit_rf))
