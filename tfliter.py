import tensorflow as tf

# Load the TensorFlow model
model = tf.saved_model.load('./efficientdet_lite2_detection_1')

# Get the concrete function
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
dimensions=[1,640,640,3]
concrete_func.inputs[0].set_shape(dimensions)
file_name=f"model.tflite-{dimensions[1]}x{dimensions[2]}"

# Convert the model
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save the TFLite model
with open(file_name, 'wb') as f:
    f.write(tflite_model)
