import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from skimage.util import random_noise
import cv2
import glob
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
import os


n_epochs = 2

# Load images
image_paths = glob.glob('Dataset_BUSI_with_GT/malignant/*.png') # Adjust as needed
image_paths = [path for path in image_paths if not path.endswith('_mask.png')]


images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

# Normalize and resize
images = [cv2.resize(img, (128, 128)) for img in images]
images = [img / 255.0 for img in images]

# Add speckle noise
noisy_images = [random_noise(img, mode='speckle') for img in images]


# Show first 5 pairs of original and noisy images
for i in range(5):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(images[i], cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(noisy_images[i], cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')

    plt.show()

X_train, X_test, y_train, y_test = train_test_split(noisy_images, images, test_size=0.2, random_state=42)


def unet_model():
    inputs = Input((128, 128, 1))
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expanding Path
    u6 = UpSampling2D(size=(2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = UpSampling2D(size=(2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = UpSampling2D(size=(2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = UpSampling2D(size=(2, 2))(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    def custom_psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    def custom_ssim(y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, max_val=1.0)


    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=[custom_psnr, custom_ssim, 'mae'])
    return model


model = unet_model()

X_train = np.array(X_train).reshape(-1, 128, 128, 1)
y_train = np.array(y_train).reshape(-1, 128, 128, 1)
X_test = np.array(X_test).reshape(-1, 128, 128, 1)
y_test = np.array(y_test).reshape(-1, 128, 128, 1)

X_train = np.expand_dims(X_train, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)

history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=32, validation_split=0.1)

X_test = np.expand_dims(X_test, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

loss, psnr, ssim, mae = model.evaluate(X_test, y_test)
evaluation_metrics = model.evaluate(X_test, y_test)
loss = evaluation_metrics[0]
mae = evaluation_metrics[3]
print('Loss:', loss)
print('Mean Absolute Error:', mae)
print('ssim:', ssim)
print('psnr:', psnr)

# Optionally, visualize the denoised images
import matplotlib.pyplot as plt

predictions = model.predict(X_test)
for i in range(5): # Show 5 examples
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title('Noisy Image')
    plt.subplot(1, 3, 2)
    plt.imshow(y_test[i].squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 3, 3)
    plt.imshow(predictions[i].squeeze(), cmap='gray')
    plt.title('Denoised Image')
    plt.show()


# Plotting loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting PSNR
plt.figure(figsize=(6, 4))
plt.plot(history.history['custom_psnr'], label='Training PSNR')
plt.plot(history.history['val_custom_psnr'], label='Validation PSNR')
plt.title('PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()
plt.show()

# Plotting SSIM
plt.figure(figsize=(6, 4))
plt.plot(history.history['custom_ssim'], label='Training SSIM')
plt.plot(history.history['val_custom_ssim'], label='Validation SSIM')
plt.title('SSIM')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.legend()
plt.show()