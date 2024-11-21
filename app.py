# from flask import Flask, render_template, Response, request
# import cv2
# import numpy as np
# import tensorflow as tf
# import winsound  # For playing a beep sound (Windows only)

# # Initialize Flask app
# app = Flask(__name__)

# # Load the pre-trained model
# model = tf.keras.models.load_model('model/fatigue_model_cnn.h5')

# # Load Haar cascade classifiers
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# # Camera variable
# camera = None

# # Function for processing and making predictions
# def process_frame(frame):
#     global model, face_cascade, eye_cascade

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#     status = "Unknown"

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_color = frame[y:y + h, x:x + w]

#         # Detect eyes within the face region
#         eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
#         if len(eyes) > 0:
#             for (ex, ey, ew, eh) in eyes:
#                 roi_eye_color = roi_color[ey:ey + eh, ex:ex + ew]
#                 try:
#                     final_image = cv2.resize(roi_eye_color, (224, 224))
#                     final_image = np.expand_dims(final_image, axis=0)
#                     final_image = final_image / 255.0

#                     # Model prediction
#                     Predictions = model.predict(final_image)
#                     predicted_class = np.argmax(Predictions, axis=1)[0]
#                     confidence = np.max(Predictions)

#                     if predicted_class == 0:  # Active
#                         status = f"Active ({confidence * 100:.2f}%)"
#                     elif predicted_class == 1:  # Sleepy
#                         status = f"Sleepy ({confidence * 100:.2f}%)"

#                 except Exception as e:
#                     print(f"Error: {e}")
#                     status = "Error"

#         # Draw face rectangle
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#     return frame, status

# # Function to generate video feed
# def generate_frames():
#     global camera
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         # Process frame
#         frame, status = process_frame(frame)

#         # Add status text
#         cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#         # Encode frame for streaming
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/start_camera', methods=['POST'])
# def start_camera():
#     global camera
#     camera = cv2.VideoCapture(0)
#     return "Camera started", 200

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/stop_camera', methods=['POST'])
# def stop_camera():
#     global camera
#     if camera is not None:
#         camera.release()
#         camera = None
#     return "Camera stopped", 200

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
# import winsound  # For playing a beep sound (Windows only)
# from playsound import playsound
 # Replace with your sound file


# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('model/fatigue_model_cnn.h5')

# Load Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Variables for consecutive sleepy frames
sleepy_count = 0
sleepy_threshold = 8
alarm_triggered = False

# Initialize camera
camera = None

def process_frame(frame):
    """
    Process the frame to detect sleepiness and trigger an alarm if necessary.
    """
    global sleepy_count, sleepy_threshold, alarm_triggered

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    status = "Unknown"

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                roi_eye_color = roi_color[ey:ey + eh, ex:ex + ew]

                # Resize and normalize the detected eye region
                try:
                    final_image = cv2.resize(roi_eye_color, (224, 224))
                    final_image = np.expand_dims(final_image, axis=0) / 255.0

                    # Make predictions
                    predictions = model.predict(final_image)
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    confidence = np.max(predictions)

                    if predicted_class == 0:  # Active
                        status = f"Active ({confidence * 100:.2f}%)"
                        sleepy_count = 0  # Reset sleepy count
                        alarm_triggered = False  # Stop the alarm
                    elif predicted_class == 1:  # Sleepy
                        status = f"Sleepy ({confidence * 100:.2f}%)"
                        sleepy_count += 1

                except Exception as e:
                    print(f"Error in processing: {e}")
                    status = "Error"

        else:
            # If no eyes detected, increment sleepy count
            status = "Sleepy"
            sleepy_count += 1

        # # Trigger alarm if sleepy threshold is reached
        # if sleepy_count >= sleepy_threshold and not alarm_triggered:
        #     playsound('alarm.wav')  # Replace with your sound file
        #     # winsound.Beep(1000, 1000)  # Beep sound (frequency 1000Hz, duration 1000ms)
        #     alarm_triggered = True  # Set alarm triggered flag
        #     sleepy_count = 0  # Reset sleepy count after alarm

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame, status

def generate_frames():
    """
    Generate frames for the video feed.
    """
    global camera
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Process the frame
        frame, status = process_frame(frame)

        # Add status text on the frame
        cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """
    Render the homepage.
    """
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """
    Start the camera.
    """
    global camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return "Failed to open the camera", 500
    return "Camera started", 200

@app.route('/video_feed')
def video_feed():
    """
    Video feed route.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """
    Stop the camera.
    """
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return "Camera stopped", 200

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False
)
