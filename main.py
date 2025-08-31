import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Load model
model = load_model("over_aug_cnn_model.h5")

# Define class labels (adjust to your dataset)
class_names =  ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']  # replace with your actual labels

# Load & preprocess an image
img_path = "pnuemonia.png"
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
preds = (model.predict(img_array)>0.08).astype(int)
pred_class = np.argmax(preds[0])
pred_label = class_names[pred_class]
print(f"Prediction: {pred_label} ({preds[0][pred_class]:.2f} confidence)")

# -------- Grad-CAM -----------
last_conv_layer = model.get_layer([layer.name for layer in model.layers if 'conv' in layer.name][-1])
grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, pred_class]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

# Overlay
img = cv2.imread(img_path)
img = cv2.resize(img, (128, 128))
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# Show results
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1,3,2)
plt.title("Heatmap")
plt.imshow(heatmap)

plt.subplot(1,3,3)
plt.title(f"Overlay - {pred_label}")
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

plt.show()
