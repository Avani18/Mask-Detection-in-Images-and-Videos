# Import the packages 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths 
import matplotlib.pyplot as plt 
import numpy as np
import argparse
import os


# Construct the argument parser and parse the argument 
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required = True, help = 'path to input dataset')
ap.add_argument('-p', '--plot', type = str, default = 'plot.png' , help = 'path to output loss/accuracy plot')
ap.add_argument('-m', '--model', type = str, default = 'mask_detector.model' , help = 'path to output face mask detector model')
args = vars(ap.parse_args())

# Initialise the initial learning rate, number of epochs to train for, batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Load and preprocess training data

# Grab list of images in dataset directory
print ("[INFO] loading images...")
imagePaths = list(paths.list_images(args['dataset']))
data = []
labels = []

# Loop over image paths
for imagePath in imagePaths:
	# Extract class label from filename	
	label = imagePath.split(os.path.sep)[-2]
	
	# Load the input image (224*224) and preprocess it
	image = load_img(imagePath, target_size = (224,224))
	# Converts image to (ht, wd, channels)
	image = img_to_array(image)
	# Preprocesses the image input so as to make it compatible with model being used 
	# Scaling the pixel intensities to range [-1, 1]
	image = preprocess_input(image)
	
	# Update the data and labels list
	data.append(image)
	labels.append(label)
	
# Convert the data and labels to Numpy arrays
data = np.array(data, dtype = 'float32')
labels = np.array(labels)

# Perform one-hot encoding on the labels 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Split the data
trainX, testX, trainY, testY = train_test_split(data, labels, test_size = 0.2, stratify = labels, random_state = 42)

# Construct the training image generator 
aug = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.15, horizontal_flip = True, fill_mode = 'nearest')

# Load MobileNetV2 network 
baseModel = MobileNetV2(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(224, 224, 3)))

# Construct head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size = (7,7))(headModel)
headModel = Flatten(name = 'Flatten')(headModel)
headModel = Dense(128, activation = 'relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation = 'softmax')(headModel)

# Place head FC model on top of base model
model = Model(inputs = baseModel.input, outputs = headModel)

# Loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# Compile our model
print ("[INFO] compiling input...")
opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

# Train the head of the network 
print ("[INFO] compiling network...")
H = model.fit(aug.flow(trainX, trainY, batch_size = BS), steps_per_epoch = len(trainX) // BS, validation_data = (testX, testY), validation_steps = len(testX) // BS, epochs = EPOCHS)

# Make predictions on testing set 
print ("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size = BS)

predIdxs = np.argmax(predIdxs, axis = 1)

print (classification_report(testY.argmax(axis = 1), predIdxs, target_names = lb.classes_))

print ("[INFO] saving mask detector model...")
model.save(args['model'], save_format = 'h5')

# Plot training loss and accuracy 

N = EPOCHS
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])






