import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

model = tf.keras.models.Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax') # dense layer has a shape of 2 as we have only 2 classes
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def trainModel():
  training_dir = "data/train"
  train_datagen = ImageDataGenerator(rescale=1.0/255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

  train_generator = train_datagen.flow_from_directory(training_dir,
                                                      batch_size=10,
                                                      target_size=(IMG_WIDTH, IMG_HEIGHT))
  validation_dir = "data/test"
  validation_datagen = ImageDataGenerator(rescale=1.0/255)
  validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                           batch_size=10,
                                                           target_size=(IMG_WIDTH, IMG_HEIGHT))
  checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

  history = model.fit_generator(train_generator,
                                epochs=epochs,
                                validation_data=validation_generator,
                                callbacks=[checkpoint])
  global acc
  acc = history.history['accuracy']
  global val_acc
  val_acc = history.history['val_accuracy']
  global loss
  loss = history.history['loss']
  global val_loss
  val_loss = history.history['val_loss']