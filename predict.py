import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("face_mask_model.keras")

img = image.load_img("data\with_mask\with_mask_23.jpg", target_size=(100, 100))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
if prediction > 0.5:
    print("No Mask 😷")
else:
    print("With Mask 😷✅")
