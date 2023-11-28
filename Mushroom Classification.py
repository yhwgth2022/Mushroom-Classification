#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import os
import shutil
import pandas as pd


# Data link: https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images

# In[2]:


new_base_dir = pathlib.Path("./Mushrooms")
new_dir = new_base_dir/"Selected_Mushrooms"


# In[3]:


def load_image(im_path):
    img = tf.io.read_file(im_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (64, 64))
    

    
for root, folder, files in os.walk(new_dir):
    for file in files:
        file_path = os.path.join(root, file)
        load_image(file_path)


# In[4]:


im_path = file_path
img = tf.io.read_file(im_path)
img = tf.image.decode_jpeg(img, channels=3)


# In[5]:


dir=str(new_dir)
get_ipython().system('ls $dir')
get_ipython().system('ls $dir/* | wc -l')


# In[6]:


jpgfiles = new_dir.glob('*/*.jpg')
jpgfiles


# In[7]:


jpgfiles = new_dir.glob('*/*.jpg')

plt.figure(figsize=(25, 10))

for i, jpg in zip(range(10), jpgfiles):
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(PIL.Image.open(str(jpg)))
    plt.axis("off")


# In[8]:


resize_layer = tf.keras.layers.Resizing(
    height = 64,
    width = 64,
    interpolation='bilinear'
)

img = np.array(PIL.Image.open(str(jpg)))
resized_img = resize_layer(img)

plt.imshow(resized_img.numpy().astype(int))


# In[9]:


plt.imshow(img)


# In[10]:


batch_size = 64
img_height = 64
img_width = 64

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
  new_dir,
  validation_split=0.1,
  subset="both",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[11]:


x, y = next(iter(train_ds))
x.shape, y.shape


# In[12]:


class_names = train_ds.class_names
print(class_names)


# In[13]:


# Initialize a dictionary to count the number of cases per class
class_counts = {class_name: 0 for class_name in class_names}

# Count the number of cases per class in the training set
for images, labels in train_ds:
    for label in labels.numpy():
        class_counts[class_names[label]] += 1

# Print the results
for class_name in class_names:
    print(f"{class_name}: {class_counts[class_name]}")


# In[14]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the data augmentation settings
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest')

# Create a new dataset using data augmentation
augmented_ds = datagen.flow_from_directory(
    new_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed=123,
    shuffle=True,
    class_mode='categorical')


# In[15]:


#Standrtize the dataset
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))


# In[16]:


nrows = 3
ncols = 3
pos = 0
import matplotlib.image as mpimg
for subfolder in os.listdir(new_dir):
    
    image_file = os.listdir(os.path.join(new_dir, subfolder))[0]
    
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pos += 1
    sp = plt.subplot(nrows, ncols, pos)

    cur_image = mpimg.imread(os.path.join(new_dir, subfolder, image_file))
    plt.imshow(cur_image)
    plt.title(subfolder)
    plt.axis('Off')


# In[17]:


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# In[18]:


data_augmentation = tf.keras.models.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
  ]
)


# In[19]:


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")


# In[20]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# In[21]:


tf.data.AUTOTUNE


# In[22]:


normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))


# In[23]:


input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))

conv1_8_3x3 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='valid', activation='relu')

maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2_8_2x2 = tf.keras.layers.Conv2D(filters=16, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension


# In[24]:


from tensorflow.keras.optimizers import Adam

model = tf.keras.models.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1_8_3x3,
        maxpool1,
        conv2_8_2x2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(3600/2, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


# In[25]:


epochs=150
batch = 80
Table = pd.DataFrame()
history = model.fit(
  train_ds,
  validation_data=val_ds,
    batch_size=batch,
  epochs=epochs
)


# In[26]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[27]:


#model_f8_w3x3 
channels1 = 8
kernel1 = 3
epochs=150

input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))
conv1 = tf.keras.layers.Conv2D(filters=channels1, kernel_size=kernel1, strides=1, padding='valid', activation='relu')
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension

model = tf.keras.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1,
        maxpool1,
        conv2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(1800/2, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


# In[28]:



history = model.fit(
  train_ds,
  validation_data=val_ds,
    batch_size=batch,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

accuracy = model.evaluate(val_ds)[1]

List_epochs = list()
List_epochs.append(epochs)
List_channel = list()
List_channel.append(channels1)
List_kernel = list()
List_kernel.append(kernel1)
Acc = list()
Acc.append(accuracy)

List_epochs = list()
List_epochs.append(epochs)
Table = pd.DataFrame({'List_channel' : List_channel,
                                'List_kernel' : List_kernel,
                                'Acc' : Acc,'epochs':List_epochs,}, 
                                columns=['List_channel','List_kernel','Acc', 'epochs'])
Table


# In[29]:


#model_f8_w5x5
channels1 = 8
kernel1 = 5

input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))
conv1 = tf.keras.layers.Conv2D(filters=channels1, kernel_size=kernel1, strides=1, padding='valid', activation='relu')
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension

model = tf.keras.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1,
        maxpool1,
        conv2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(1568/2, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


# In[30]:


epochs=150
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

accuracy = model.evaluate(val_ds)[1]


List_channel.append(channels1)

List_kernel.append(kernel1)

Acc.append(accuracy)

List_epochs.append(epochs)

Table = pd.DataFrame({'List_channel' : List_channel,
                                'List_kernel' : List_kernel,
                                'Acc' : Acc,'epochs':List_epochs}, 
                                columns=['List_channel','List_kernel','Acc', 'epochs'])
Table


# In[31]:


#model_f16_w3x3
channels1 = 16
kernel1 = 3

input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))
conv1 = tf.keras.layers.Conv2D(filters=channels1, kernel_size=kernel1, strides=1, padding='valid', activation='relu')
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension

model = tf.keras.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1,
        maxpool1,
        conv2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(1800/2, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


# In[32]:


epochs=150
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

accuracy = model.evaluate(val_ds)[1]


List_channel.append(channels1)

List_kernel.append(kernel1)

Acc.append(accuracy)

List_epochs.append(epochs)

Table = pd.DataFrame({'List_channel' : List_channel,
                                'List_kernel' : List_kernel,
                                'Acc' : Acc,'epochs':List_epochs}, 
                                columns=['List_channel','List_kernel','Acc', 'epochs'])
Table


# In[33]:


#model_f16_w5x5
channels1 = 16
kernel1 = 5

input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))
conv1 = tf.keras.layers.Conv2D(filters=channels1, kernel_size=kernel1, strides=1, padding='valid', activation='relu')
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension

model = tf.keras.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1,
        maxpool1,
        conv2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(1568/2, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()


# In[34]:


epochs=150
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

accuracy = model.evaluate(val_ds)[1]


List_channel.append(channels1)

List_kernel.append(kernel1)

Acc.append(accuracy)

List_epochs.append(epochs)

Table = pd.DataFrame({'List_channel' : List_channel,
                                'List_kernel' : List_kernel,
                                'Acc' : Acc,'epochs':List_epochs}, 
                                columns=['List_channel','List_kernel','Acc', 'epochs'])
Table


# In[35]:


#model_f24_w3x3
channels1 = 24
kernel1 = 3

input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))
conv1 = tf.keras.layers.Conv2D(filters=channels1, kernel_size=kernel1, strides=1, padding='valid', activation='relu')
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension

model = tf.keras.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1,
        maxpool1,
        conv2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(1800/2, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


# In[36]:


epochs=150
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

accuracy = model.evaluate(val_ds)[1]


List_channel.append(channels1)

List_kernel.append(kernel1)

Acc.append(accuracy)

List_epochs.append(epochs)

Table = pd.DataFrame({'List_channel' : List_channel,
                                'List_kernel' : List_kernel,
                                'Acc' : Acc,'epochs':List_epochs}, 
                                columns=['List_channel','List_kernel','Acc', 'epochs'])
Table


# In[37]:


#model_f24_w5x5
channels1 = 24
kernel1 = 5

input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))
conv1 = tf.keras.layers.Conv2D(filters=channels1, kernel_size=kernel1, strides=1, padding='valid', activation='relu')
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension

model = tf.keras.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1,
        maxpool1,
        conv2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(1568/2, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


# In[38]:


epochs=150
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

accuracy = model.evaluate(val_ds)[1]


List_channel.append(channels1)

List_kernel.append(kernel1)

Acc.append(accuracy)

List_epochs.append(epochs)

Table = pd.DataFrame({'List_channel' : List_channel,
                                'List_kernel' : List_kernel,
                                'Acc' : Acc,'epochs':List_epochs}, 
                                columns=['List_channel','List_kernel','Acc', 'epochs'])
Table


# # #Best performance analysis
# ###Channel:16, Kernel:3

# In[39]:


#model_f16_w3x3
channels1 = 16
kernel1 = 3

input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))
conv1 = tf.keras.layers.Conv2D(filters=channels1, kernel_size=kernel1, strides=1, padding='valid', activation='relu')
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension

best_model = tf.keras.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1,
        maxpool1,
        conv2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(1800/2, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

best_model.summary()


# In[40]:


epochs=300
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

accuracy = model.evaluate(val_ds)[1]


List_channel.append(channels1)

List_kernel.append(kernel1)

Acc.append(accuracy)

List_epochs.append(epochs)

Best_Table = pd.DataFrame({'List_channel' : List_channel,
                                'List_kernel' : List_kernel,
                                'Acc' : Acc,'epochs':List_epochs}, 
                                columns=['List_channel','List_kernel','Acc', 'epochs'])
Best_Table


# In[41]:


epochs=100
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

accuracy = model.evaluate(val_ds)[1]


List_channel.append(channels1)

List_kernel.append(kernel1)

Acc.append(accuracy)

List_epochs.append(epochs)

Best_Table = pd.DataFrame({'List_channel' : List_channel,
                                'List_kernel' : List_kernel,
                                'Acc' : Acc,'epochs':List_epochs}, 
                                columns=['List_channel','List_kernel','Acc', 'epochs'])
Best_Table


# In[42]:


Best_Table


# In[43]:


epochs=150
history = best_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

from sklearn.metrics import confusion_matrix

# Get the true labels and predictions for the test set
test_labels = np.concatenate([y for x, y in val_ds], axis=0)
y_pred = model.predict(val_ds, verbose=1)
pred_labels = np.argmax(y_pred, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(test_labels, pred_labels)

# Normalize the confusion matrix per class
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Print the confusion matrix as percentages per class
print("Confusion matrix (as percentages per class):")
print(np.round(cm_norm * 100, decimals=2))


# In[44]:


# Extract all the Zn configurations from the test set
Zn_list = []
combined_ds = val_ds.concatenate(val_ds)
for x, y in combined_ds:
    Zn = best_model.layers[6](x)  # assuming flatten layer is at index 6
    Zn_list.append(Zn.numpy())
Zn = np.concatenate(Zn_list, axis=0)


# In[45]:


# Standardize the extracted configurations
Zn_std = (Zn - np.mean(Zn, axis=0)) / np.std(Zn, axis=0)

# Compute the covariance matrix
cov_mat = np.cov(Zn_std, rowvar=False)

# Compute the eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Sort the eigenvalues in descending order and calculate the cumulative sum
eig_vals_sorted = np.sort(eig_vals)[::-1]
cumsum = np.cumsum(eig_vals_sorted)
cumsum_percent = cumsum / cumsum[-1] * 100

# Determine the number of principal components required to explain 99% of the variance
n_components = np.argmax(cumsum_percent > 99) + 1

print(f"Number of principal components required to explain 99% of the variance: {n_components}")


# In[46]:


#model_f16_w3x3
channels1 = 16
kernel1 = 3

input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))
conv1 = tf.keras.layers.Conv2D(filters=channels1, kernel_size=kernel1, strides=1, padding='valid', activation='relu')
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension

best_model = tf.keras.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1,
        maxpool1,
        conv2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(n_components, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

best_model.summary()


epochs=150
history = best_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



accuracy = best_model.evaluate(val_ds)[1]

Hid = list()
Hid.append('R')

Accu = list()
Accu.append(accuracy)



H_Table = pd.DataFrame({'h' : Hid,
                                'Accuracy' : Accu}, 
                                columns=['h','Accuracy'])
H_Table


# In[47]:


#model_f16_w3x3
channels1 = 16
kernel1 = 3

input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))
conv1 = tf.keras.layers.Conv2D(filters=channels1, kernel_size=kernel1, strides=1, padding='valid', activation='relu')
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension

best_model = tf.keras.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1,
        maxpool1,
        conv2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(2*n_components, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

best_model.summary()


epochs=150
history = best_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



accuracy = best_model.evaluate(val_ds)[1]


Hid.append('2R')


Accu.append(accuracy)



H_Table = pd.DataFrame({'h' : Hid,
                                'Accuracy' : Accu}, 
                                columns=['h','Accuracy'])
H_Table


# In[48]:


#model_f16_w3x3
channels1 = 16
kernel1 = 3

input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))
conv1 = tf.keras.layers.Conv2D(filters=channels1, kernel_size=kernel1, strides=1, padding='valid', activation='relu')
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension

best_model = tf.keras.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1,
        maxpool1,
        conv2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(1800, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

best_model.summary()


epochs=150
history = best_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



accuracy = best_model.evaluate(val_ds)[1]


Hid.append('F/2')


Accu.append(accuracy)



H_Table = pd.DataFrame({'h' : Hid,
                                'Accuracy' : Accu}, 
                                columns=['h','Accuracy'])
H_Table


# In[49]:


#model_f16_w3x3
channels1 = 16
kernel1 = 3

input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))
conv1 = tf.keras.layers.Conv2D(filters=channels1, kernel_size=kernel1, strides=1, padding='valid', activation='relu')
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension

best_model = tf.keras.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1,
        maxpool1,
        conv2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(2*n_components, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

best_model.summary()


epochs=150
history = best_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# In[50]:


# Get the true labels and predictions for the test set
test_labels = np.concatenate([y for x, y in val_ds], axis=0)
y_pred = best_model.predict(val_ds, verbose=1)
pred_labels = np.argmax(y_pred, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(test_labels, pred_labels)

# Normalize the confusion matrix per class
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Print the confusion matrix as percentages per class
print("Confusion matrix (as percentages per class):")
print(np.round(cm_norm * 100, decimals=2))


# In[51]:


#CL3 CL7
class_names = ['Boletus', 'Lactarius']

# Define a function to filter out images not in the selected classes
def filter_classes(x, y):
    mask = tf.logical_or(tf.equal(y, 2), tf.equal(y, 6))
    return x[mask], y[mask]

# Apply the filter to the training set
train_ds_3_7 = train_ds.map(filter_classes)

# Apply the filter to the validation set
val_ds_3_7 = val_ds.map(filter_classes)


# In[52]:


#model_f16_w3x3
channels1 = 16
kernel1 = 3

input_layer = tf.keras.layers.InputLayer(input_shape=(img_height,img_width,3))
conv1 = tf.keras.layers.Conv2D(filters=channels1, kernel_size=kernel1, strides=1, padding='valid', activation='relu')
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='valid', activation='relu')
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
flatten = tf.keras.layers.Flatten()
# Get flattened layer dimension

best_model = tf.keras.Sequential(
    [
        data_augmentation,
        input_layer,
        conv1,
        maxpool1,
        conv2,
        maxpool2,
        flatten,
        tf.keras.layers.Dense(2*n_components, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ]
)
best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

best_model.summary()


epochs=150
history = best_model.fit(
  train_ds_3_7,
  validation_data=val_ds_3_7,
  epochs=epochs
)


# In[71]:


# Get the true labels and predictions for the test set
test_labels = np.concatenate([y for x, y in val_ds_3_7], axis=0)
y_pred = best_model.predict(val_ds_3_7, verbose=1)
pred_labels = np.argmax(y_pred, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(test_labels, pred_labels)

# Normalize the confusion matrix per class
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Print the confusion matrix as percentages per class
print("Confusion matrix (as percentages per class):")
print(np.round(cm_norm * 100, decimals=2))


# In[68]:


pred_labels.shape,y_pred.shape


# In[72]:



# Compute true positive and false positive rates
tn, fp, fn, tp = cm.ravel()
tpr[i] = tp / (tp + fn)
fpr[i] = fp / (fp + tn)

# Compute area under the curve (AUC)
auc_score = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {auc_score:.3f})')
plt.show()


# In[58]:




