# Paso 1: Configurar el entorno
!pip install opencv-python
!pip install opencv-python-headless
!pip install requests

# Paso 2: Cargar la imagen
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt


# Paso 3: Obtener la imagen a través de requests
# URL de la imagen
url = "https://variety.com/wp-content/uploads/2022/08/Jonah-Hill.jpg?w=1000&h=563&crop=1"
response = requests.get(url)

# Comprobar si la solicitud fue exitosa
if response.status_code == 200:
    image = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Verificar si la imagen se cargó correctamente
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB

        # Mostrar la imagen original
        plt.imshow(image)
        plt.axis('off')
        plt.title('Imagen Original')
        plt.show()
    else:
        print("Error: La imagen no se cargó correctamente.")
else:
    print(f"Error al cargar la imagen. Código de estado: {response.status_code}")
    
# Paso 4: Detectar caras
# Cargar el clasificador en cascada
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Detectar caras
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# Dibujar rectángulos alrededor de las caras detectadas
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Mostrar la imagen con las caras detectadas
plt.figure(figsize=(10, 6))  # Ajustar el tamaño de la figura
plt.imshow(image)
plt.axis('on')  # No mostrar ejes
plt.title('Detección de Caras', fontsize=16)  # Cambiar el tamaño del texto del título
plt.show()

# Paso 5: Detectar ojos
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Detectar ojos en la región de la cara
for (x, y, w, h) in faces:
    roi_gray = gray_image[y:y+h, x:x+w]  # Región de interés para la cara
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # Dibujar rectángulos alrededor de los ojos detectados
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Rectángulo verde para ojos

# Mostrar la imagen con las caras y ojos detectados
plt.figure(figsize=(10, 6))  # Ajustar el tamaño de la figura
plt.imshow(image)
plt.axis('off')  # No mostrar ejes
plt.title('Detección de Caras y Ojos', fontsize=16)  # Cambiar el tamaño del texto del título
plt.show()