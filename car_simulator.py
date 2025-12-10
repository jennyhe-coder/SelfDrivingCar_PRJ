import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# data preprocessing function 
def preproc_img(img):
    # crop the road area from the image 
    img = img[60: 135, :, :]

    # convert the image to YUV color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # resize the image to 200 x 66 pixels used by the NVIDIA model
    img = cv2.resize(img, (200, 66))

    # apply Gaussian blur filter
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # apply normalization to this range [-1, 1]
    img = img / 127.5 - 1.0

    return img

# custom batch generator  
def custom_batch_generator(image_paths, angles, batch_size, datagen=None, training=True):
    num_samples = len(image_paths)
    image_paths = np.array(image_paths)
    angles = np.array(angles)

    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start in range(0, num_samples, batch_size):
            batch_indices = indices[start:start+batch_size]

            batch_imgs = []
            batch_angles = []

            for idx in batch_indices:
                path = image_paths[idx]

                # load BGR image using cv2
                img = cv2.imread(path)

                # preprocessing
                img = preproc_img(img)

                batch_imgs.append(img)
                batch_angles.append(angles[idx])

            batch_imgs = np.array(batch_imgs)
            batch_angles = np.array(batch_angles)

            # apply augmentation only during training
            if training and datagen:
                # datagen.flow returns (X, y)
                batch_imgs_aug = next(datagen.flow(
                    batch_imgs,
                    batch_angles,
                    batch_size=batch_size,
                    shuffle=False
                ))[0]
                batch_imgs = batch_imgs_aug

            yield batch_imgs, batch_angles

# load dataset 
df = pd.read_csv('driving_log.csv')

# get the last column which is the steering angle
steering_angle = df.iloc[:, 3]

# get the images for the road area
road_image = df.iloc[:, 0]

# split the data (set random state as 42 for reproducibility)
x_train, x_test, y_train, y_test = train_test_split(road_image, steering_angle, test_size=0.2, random_state=42)

# improve generalization of the model
# Apply data augmentation techniques to only the training dataset 
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# create custom generators
batch_size = 25

train_generator = custom_batch_generator(
    x_train, 
    y_train,
    batch_size=batch_size,
    datagen=datagen, 
    training=True
)

# For test/validation data we need a separate generator w/o augmentation
test_datagen = custom_batch_generator(
    x_test, 
    y_test,
    batch_size=batch_size,
    datagen=None, 
    training=False
)

# Build the neural network
model = Sequential()

model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (3, 3), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train model 
# get batch size
steps = len(x_train)
validation_steps = len(x_test)

trained_model = model.fit(
    train_generator,
    steps_per_epoch=steps,
    validation_data=test_datagen,
    validation_steps=validation_steps,
    epochs=5,
    verbose=1
)

# get some test samples to plot
x_test_sample, y_test_sample = next(test_datagen)
y_pred = model.predict(x_test_sample)

# plot a chart for the predicted steering angles  
plt.figure(figsize=(10, 5))
plt.plot(y_test_sample, label='Actual Steering Angles')
plt.plot(y_pred, label='Predicted Steering Angles')
plt.legend()
plt.title("Predicted vs Actual Steering Angles")
plt.ylabel("Steering Angle")
plt.grid()
plt.show()
