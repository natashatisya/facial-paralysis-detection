from flask import Flask, render_template, request, session, Response, url_for, jsonify
from flask_socketio import SocketIO, emit
import os
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'f@c1aL_P@r@lyS1s@FYP!' # Secret key for Flask session management

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the frame and make predictions
def predict_webcam_frame(frame, model):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) corresponding to the face
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face ROI to the input size expected by the model
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = image.img_to_array(face_roi)
        face_roi = face_roi / 255.0  # Normalize the image
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        # Make predictions on the face ROI
        result = model.predict(face_roi)

        # Extract the confidence or probability score
        confidence = result[0][0] if result[0][0] > 0.5 else 1 - result[0][0]

        # Draw a rectangle around the detected face with green color
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the classification result and confidence on the frame with green color
        text_color = (0, 255, 0)
        if result > 0.5:
            classification_result = f"Stroke Face ({int(confidence * 100)}%)"
        else:
            classification_result = f"Normal Face ({int(confidence * 100)}%)"

        cv2.putText(frame, classification_result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    return frame

# Function to create and compile the InceptionResNetV2 model
def create_inceptionresnetv2_model():
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    custom_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=custom_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Load the pre-trained model
model_directory = os.path.join(os.getcwd(), 'Model')
model_filename = 'inception_resnetv2_model.h5'
model_path = os.path.join(model_directory, model_filename)
loaded_model = load_model(model_path)

# Modify the predict_single_image function
def predict_single_image(file, model):
    try:
        img = image.load_img(file, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        
        result = model.predict(img)
        probability = result[0][0]

        if probability > 0.5:
            classification_result = "Stroke Face"
        else:
            classification_result = "Normal Face"

        return img, classification_result

    except Exception as e:
        print("Exception during prediction:", str(e), flush=True)
        return None, "Error during prediction"

# Flask route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Function to generate frames for video feed
def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        result_frame = predict_webcam_frame(frame, loaded_model)

        _, buffer = cv2.imencode('.jpg', result_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Flask route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route for the 'about' page
@app.route("/about")
def about():
    return render_template('about.html')

# Flask route for the 'upload' page
@app.route("/upload")
def upload():
    return render_template('upload.html')

# Flask route for the 'preview' page
@app.route("/preview")
def preview():
    return render_template('previewimage.html')

# Flask route for predicting uploaded image
@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('upload.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', error='No selected file')
    if file:
        upload_folder = os.path.join(app.root_path, 'static', 'uploads')
        file_path = os.path.join(upload_folder, file.filename)
        os.makedirs(upload_folder, exist_ok=True)
        file.save(file_path)
        session['file_path'] = file_path

        return render_template('previewimage.html')

# Flask route for the 'prediction' page
@app.route("/prediction")
def prediction():
    file_path=session.get("file_path",None)
    img, result = predict_single_image(file_path, loaded_model)
    print("Result for Prediction:", result, flush=True)
    return render_template('prediction.html', img=file_path, result=result)

# Flask route for the 'result' page
@app.route("/result")
def result():
    return render_template('result.html')

# Run the Flask app if this script is executed
if __name__ == '__main__':
    app.run(debug=True)