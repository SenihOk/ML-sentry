import numpy as np
import cv2
from screeninfo import get_monitors
import tensorflow.lite as tf
import time
import math

# Load the TFLite model
interpreter = tf.Interpreter(model_path='./ssd_mobilenet_v3_small_coco_2020_01_14/model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#get window dimensions
monitor = get_monitors()[0]
# screen_width = monitor.width
# screen_height = monitor.height
window_width = int(monitor.width * 0.5)
window_height = int(monitor.height * 0.5)
# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 896)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 896)


# Initialize the time and frames counter
start_time = time.time()
num_frames = 0

##prints debugging details
# print(input_details)
# print(output_details)


class Intruder:
    def __init__(self, center, confidence):
        self.center = center
        self.radius = 10
        self.confidence = confidence
        self.present = True
    
    def update(self, center, confidence):
        distance = math.sqrt((center[0] - self.center[0]) ** 2 + (center[1] - self.center[1]) ** 2)
        if (distance < self.radius):
            self.present = False
        else:
            self.present = True
            self.center = center
        self.confidence = confidence

    def checkPresence(self):
        return self.present


while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    #resize video frame
    frame = cv2.resize(frame, (320, 320))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(frame_rgb, 0).astype(np.uint8)

    # Calculate FPS
    num_frames += 1
    elapsed_time = time.time() - start_time
    fps = num_frames / elapsed_time
    
    # Convert the frame to a tensor
    # input_tensor = np.expand_dims(frame, 0).astype(np.uint8)

    # Set the tensor as input to the model
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run the model
    interpreter.invoke()
    
    # Load the output tensors
    boxes_tensor = interpreter.get_tensor(output_details[0]['index'])
    classes_tensor = interpreter.get_tensor(output_details[1]['index'])
    scores_tensor = interpreter.get_tensor(output_details[2]['index'])

    # Extract the tensors
    boxes = boxes_tensor[0]
    classes = classes_tensor[0]
    scores = scores_tensor[0]

    #redraw frame for user screen
    frame = cv2.resize(frame, (window_width, window_width))

    # Filter out detections that are not persons or have a low score
    person_indices = np.logical_and(classes == 0, scores > 0.5)
    # Draw bounding boxes and scores for detected persons
    for i in np.where(person_indices)[0]:
        # print('person detected!')
        box = boxes[i]
        score = scores[i]

        # Convert box coordinates to pixel coordinates
        y_min, x_min, y_max, x_max = box
        x_min = int(x_min * frame.shape[1])
        x_max = int(x_max * frame.shape[1])
        y_min = int(y_min * frame.shape[0])
        y_max = int(y_max * frame.shape[0])

        box_width = x_max - x_min
        box_height = y_max - y_min
        aspect_ratio = box_width / box_height

        # Define the aspect ratio for a "tall" bounding box (where the center would be 35% down from the top)
        aspect_ratio_tall = 0.25  # Width is a quarter of the height

        # Define the aspect ratio for a "wide" bounding box (where the center would be 10% down from the top)
        aspect_ratio_wide = 4.0  # Width is four times the height

        # Clamp the aspect ratio to be within the defined range
        aspect_ratio = max(min(aspect_ratio, aspect_ratio_wide), aspect_ratio_tall)

        # Map the aspect ratio to the range [0, 1]
        mapped_aspect_ratio = (aspect_ratio - aspect_ratio_tall) / (aspect_ratio_wide - aspect_ratio_tall)

        # Define the center percentages for the tall and wide bounding boxes
        center_percentage_tall = 0.35
        center_percentage_wide = 0.10

        # Linearly interpolate the center percentage based on the mapped aspect ratio
        center_percentage = (1 - mapped_aspect_ratio) * center_percentage_tall + mapped_aspect_ratio * center_percentage_wide

        # Calculate the center coordinates
        y_center = y_min + int(box_height * center_percentage)
        x_center = int((x_min + x_max) / 2)

        box_center = (x_center, y_center)



        # Draw the bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        #draw center of box
        cv2.circle(frame, (box_center), 5, (10, 10, 250), -1)

        # Draw the score
        cv2.putText(frame, f'{score:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
       
        # Draw FPS on the output frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
