import cv2
import mediapipe as mp

# Inicializar mediapipe face detection y drawing
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Iniciar captura de video
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        
        # Dibujar detecciones
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Recuadro para la cara

                # Usar un modelo de landmarks para detectar los ojos
                face_landmarks = detection.location_data.relative_keypoints
                left_eye = face_landmarks[0]  # Ojo izquierdo
                right_eye = face_landmarks[1]  # Ojo derecho
                
                # Calcular las posiciones de los ojos
                left_eye_x, left_eye_y = int(left_eye.x * iw), int(left_eye.y * ih)
                right_eye_x, right_eye_y = int(right_eye.x * iw), int(right_eye.y * ih)

                # Definir el tamaño del rectángulo para los ojos
                eye_width, eye_height = 20, 20  # Ajusta el tamaño según sea necesario

                # Dibujar recuadros alrededor de los ojos
                cv2.rectangle(frame, (left_eye_x - eye_width // 2, left_eye_y - eye_height // 2),
                              (left_eye_x + eye_width // 2, left_eye_y + eye_height // 2), (255, 0, 0), 2)  # Ojo izquierdo
                cv2.rectangle(frame, (right_eye_x - eye_width // 2, right_eye_y - eye_height // 2),
                              (right_eye_x + eye_width // 2, right_eye_y + eye_height // 2), (255, 0, 0), 2)  # Ojo derecho

        cv2.imshow('Mediapipe Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()