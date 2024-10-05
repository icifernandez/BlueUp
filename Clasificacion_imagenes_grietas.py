import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Rutas de las carpetas donde están las imágenes
positive_folder = "C:/Users/icife/OneDrive/Escritorio/Prueba codigo/Positive"
negative_folder = "C:/Users/icife/OneDrive/Escritorio/Prueba codigo/Negative"
test_folder = "C:/Users/icife/OneDrive/Escritorio/Prueba codigo/Test"  # Carpeta para nuevas imágenes

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

# Verificar el balance de clases
print("Número de imágenes con grietas:", len(positive_images))
print("Número de imágenes sin grietas:", len(negative_images))

# Combinar ambos conjuntos de imágenes y etiquetas
X = np.array(positive_images + negative_images)  # Concatenar imágenes
Y = np.array(positive_labels + negative_labels)  # Concatenar etiquetas

# Aplanar las imágenes para pasarlas al modelo (de 64x64 a 4096 valores)
X_flat = X.reshape(len(X), -1)

# Normalización de las imágenes
X_flat = X_flat.astype('float32') / 255.0

# Dividir en conjuntos de entrenamiento y prueba
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_flat, Y, test_size=0.2, random_state=42)

# Crear el modelo Random Forest con más estimadores
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

# Ajustar el modelo con el conjunto de entrenamiento
rf_model.fit(Xtrain, Ytrain)

# Realizar predicciones en el conjunto de prueba
yfit_rf = rf_model.predict(Xtest)

# Evaluar el rendimiento
print("Accuracy (Random Forest):", accuracy_score(Ytest, yfit_rf))
print("Classification Report (Random Forest):\n", classification_report(Ytest, yfit_rf))

# Visualizar algunas predicciones del conjunto de prueba
for i in range(10):  # Mostrar 5 imágenes de prueba con sus predicciones
    plt.imshow(Xtest[i].reshape(64, 64), cmap='gray')
    plt.title(f"Predicción: {'Grieta' if yfit_rf[i] == 1 else 'Sin Grieta'}")
    plt.axis('off')  # Sin ejes
    plt.show()

# Función para clasificar y mostrar una nueva imagen
def classify_new_image(image_path, model):
    # Cargar y procesar la imagen
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Cargar en escala de grises
    img_resized = cv2.resize(img, (64, 64))  # Redimensionar a 64x64 píxeles
    img_flattened = img_resized.flatten().astype('float32') / 255.0  # Normalizar la imagen

    # Realizar la predicción
    prediction = model.predict([img_flattened])

    # Mostrar la imagen y el resultado
    plt.imshow(img_resized, cmap='gray')
    plt.axis('off')  # Sin ejes

    if prediction[0] == 1:
        plt.title("Clasificación: Grieta", color="red")  # Grieta
    else:
        plt.title("Clasificación: Sin Grieta", color="green")  # Sin grieta
    
    plt.show()

# Clasificar imágenes desde la carpeta Test
for filename in os.listdir(test_folder):
    test_image_path = os.path.join(test_folder, filename)
    classify_new_image(test_image_path, rf_model)

