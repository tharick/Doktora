# Ubuntu 22.04, Debian 12 işletim sistemleri için önce sırasıyla aşağıdaki komutlar çalıştıırlmalı
# sudo apt install python3.12-venv
# pip install gradio tensorflow


import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Modeli yükle
model = tf.keras.models.load_model('dog_skin_model.h5')

# Sınıf isimleri
class_names = ['Dermatitis', 'Healthy', 'Mange', 'Ringworm']

# Görseli alıp tahmin yapacak fonksiyon
def predict_image(img):
  try:
    img = img.resize((224, 224))  # Modelin beklediği boyut
    img_array = image.img_to_array(img) / 255.0  # Normalize et
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekle
    predictions = model.predict(img_array)[0]
    prediction_dict = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return prediction_dict
  except Exception as e:
    return {"Error": str(e)}

# Gradio arayüzünü başlat
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Dog Skin Disease Classifier",
    description="Upload an image of a dog's skin to predict possible skin diseases using a deep learning model."
)

interface.launch()
