import numpy as np
import cv2
# from screeninfo import get_monitors
import tensorflow.lite as tf

# Load the TFLite model
interpreter = tf.Interpreter(model_path='./lite-model2.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# #get window dimensions
# monitor = get_monitors()[0]

# window_width = int(monitor.width * 0.5)
# window_height = int(monitor.height * 0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 896)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 896)


# print(input_details[0]['shape'])

##prints debugging details
# print(input_details)
# print(output_details)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    #resize video frame
    frame = cv2.resize(frame, (448, 448))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(frame_rgb, 0).astype(np.uint8)

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
    # frame = cv2.resize(frame, (window_width, window_width))

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

        #get centerbpoint of box
        box_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)

        # print(f"x coord is {box_center[0]} and y coordinate is {box_center[1]}")

        center_area = (210,238)
        midpoint = 224
        coord =  - 180


        #THIS IS WHERE THE SERVO CODE NEEDS TO GO LATER
        if ((box_center[0] < center_area[0]) or (box_center[0] > center_area[1])):
            print(f"box is not center!!!")
        else:
            print(f"box centered")


        # Draw the bounding box
        # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw the score
        # cv2.putText(frame, f'{score:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    # Display the frame
    # cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
