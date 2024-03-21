import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the model architecture from JSON
with open('model_a.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the model weights
model = model_from_json(loaded_model_json)
model.load_weights('model_weights.h5')

# Define the emotions dictionary
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Initialize variables for smoothing predictions
smoothed_prediction = None
smoothing_factor = 0.5  # Adjust the smoothing factor based on your preference

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        
        # Extract the region of interest (ROI) and preprocess it for prediction
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        
        # Make a prediction using the loaded model
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        
        # Apply smoothing to the prediction
        if smoothed_prediction is None:
            smoothed_prediction = maxindex
        else:
            smoothed_prediction = int(smoothing_factor * maxindex + (1 - smoothing_factor) * smoothed_prediction)
        
        # Display the smoothed predicted emotion
        cv2.putText(frame, emotion_dict[smoothed_prediction], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

