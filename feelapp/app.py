from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
import base64
import mediapipe as mp

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar el modelo de emociones
try:
    model = tf.keras.models.load_model('modelo_xception_emociones_v2.keras')
    print("Modelo de emociones cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# Inicializar Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Ruta principal para renderizar la página web
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar emociones detectadas desde imagen
@app.route('/capture_emotion', methods=['POST'])
def capture_emotion():
    if not model:
        return jsonify({'error': 'El modelo de emociones no está disponible.'})

    data = request.get_json()
    image_data = data['image'].split(",")[1]
    print("Datos de la imagen recibidos:", len(image_data))  # Log de depuración

    try:
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print("Tamaño de la imagen:", img.shape)  # Log de depuración
    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {e}'})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(gray)

    if not results.detections:
        print("No se detectó ningún rostro.")  # Log de depuración
        return jsonify({'emotion': 'No se detectó ningún rostro'})

    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        rostro = img[y:y+h, x:x+w]
        print("Rostro detectado en coordenadas:", x, y, w, h)  # Log de depuración
        break

    rostro = cv2.resize(rostro, (224, 224))
    rostro = rostro.astype('float32') / 255.0
    rostro = np.expand_dims(rostro, axis=0)

    predicciones = model.predict(rostro)
    print("Predicciones del modelo:", predicciones)  # Log de depuración
    clase_predicha = np.argmax(predicciones)
    emotionsList = ['Alegria', 'Calma', 'Miedo', 'Tristeza']
    emocion_predicha = emotionsList[clase_predicha] if clase_predicha < len(emotionsList) else "Emoción desconocida"

    print("Emoción predicha:", emocion_predicha)  # Log de depuración
    return jsonify({'emotion': emocion_predicha})

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)