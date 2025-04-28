# 📦 Gerekli kütüphaneleri import et (bir kere çalıştırdıysan gerekmez)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import pandas as pd

# 📂 Eğitim ve doğrulama dizinleri
train_dir = '/content/dog_dataset/train'
val_dir = '/content/dog_dataset/valid'

# 📈 Veri artırma ve ölçekleme
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# 🧠 Modeli oluştur
mobilenet = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
mobilenet.trainable = False

model = models.Sequential([
    mobilenet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# ⚙️ Derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🚀 Eğitimi başlat
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# 💾 Eğitim sonuçlarını kaydet
history_df = pd.DataFrame(history.history)
history_df.to_csv('/content/history.csv', index=False)
print("history.csv başarıyla kaydedildi!")

# 💾 Modeli kaydet
model.save('/content/dog_skin_model.h5')
print("dog_skin_model.h5 başarıyla kaydedildi!")
