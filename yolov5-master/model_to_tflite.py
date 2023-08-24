import tensorflow as tf

saved_model_dir = 'yolov5-master/runs/train/exp2/weights/last.pt'
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

model_path = 'yolov5-master/runs/train/exp2/weights/last.ftlite'
# Save the model.
with open(model_path, 'wb') as f:
  f.write(tflite_model)