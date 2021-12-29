#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# tf.config.list_physical_devices('GPU') 
from tensorflow import keras
from tensorflow.keras import layers, activations
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/'
import librosa
import librosa.display
import scipy
import IPython.display as ipd
from scipy import signal
from scipy.io import wavfile
import shutil
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path


# In[2]:


### Unzipping the sound files from train & test folders
# shutil.unpack_archive("train.zip", "")
# shutil.unpack_archive("test.zip", "")


# In[3]:


len(os.listdir("./test_spectrograms/"))
# Open the .csv file with all labels
# df = pd.read_csv('./train.csv')
# train 10675
# test 


# In[4]:


# Show labels
# df.head()


# In[5]:


# Appending .jpg string to every clip_name as sounds 
# will be transformed to spectrograms
def append_ext(fn):
    return fn + ".jpg"

traindf=pd.read_csv("./train.csv",dtype=str)
testdf=pd.read_csv("./sample_submission.csv",dtype=str)
traindf["clip_name"]=traindf["clip_name"].apply(append_ext)
testdf["clip_name"]=testdf["clip_name"].apply(append_ext)


# In[6]:


# checking if the image extension has been appended
# traindf.head()

# Converting each class name label in a column to a list
traindf['label']=traindf['label'].apply(lambda x:str(x).split(','))
# print(traindf['label'])
# print(testdf["clip_name"])


# In[7]:


### Visualising one audio clip
# audio_fpath = 'D:/03_AAU_Masters/03_3rd_semester/01_project/whale_data/data/sample/yes/train6.aiff'
# spectrograms_path = 'D:/03_AAU_Masters/03_3rd_semester/01_project/whale_data/data/train_spectrograms/'
# audio_clips = os.listdir(audio_fpath)

# FIG_SIZE = (8,6)
# signal, sample_rate = librosa.load(audio_fpath,sr = None) # your data sample rate is 2000 Hz, retrieved from the files
# plt.figure(figsize=FIG_SIZE)
# librosa.display.waveplot(signal, sample_rate, alpha=0.4)
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Waveform")
# plt.show()


# In[8]:


# perform Fourier transform
# fft = np.fft.fft(signal)
# # calculate abs values on complex numbers to get magnitude
# spectrum = np.abs(fft)
# # create frequency variable
# f = np.linspace(0, sample_rate, len(spectrum))
# # take half of the spectrum and frequency
# left_spectrum = spectrum[:int(len(spectrum)/2)]
# left_f = f[:int(len(spectrum)/2)]
# # plot spectrum
# plt.figure(figsize=FIG_SIZE)
# plt.plot(left_f, left_spectrum, alpha=0.4)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.title("Power spectrum")
# plt.show()
# Check, the frequencies go from 0 to FS/2 = 1000.


# In[9]:


# STFT -> spectrogram
# Figs 3 & 4 say All spectrograms used a sample rate of 250 Hz, 256 point FFT with 85% overlap.
# Your FS = 2000 Hz, 8 times higher
# hop_length = 128 # in num. of samples
# n_fft = 2048 # window in num. of samples

# # calculate duration hop length and window in seconds
# hop_length_duration = float(hop_length)/sample_rate
# n_fft_duration = float(n_fft)/sample_rate

# print("STFT hop length duration is: {}s".format(hop_length_duration))
# print("STFT window duration is: {}s".format(n_fft_duration))

# # perform stft
# stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

# # calculate abs values on complex numbers to get magnitude
# spectrogram = np.abs(stft)

# # display spectrogram
# plt.figure(figsize=FIG_SIZE)
# librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.title("Spectrogram")
# plt.show()


# In[10]:


# apply logarithm to cast amplitude to Decibels
# log_spectrogram = librosa.amplitude_to_db(spectrogram)

# plt.figure(figsize=FIG_SIZE)
# librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar(format="%+2.0f dB")
# plt.title("Spectrogram (dB)")
# plt.show()


# In[11]:


# MFCCs
# extract 13 MFCCs
# MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft,
# hop_length=hop_length, n_mfcc=39)
# # display MFCCs
# plt.figure(figsize=FIG_SIZE)
# librosa.display.specshow(MFCCs, sr=sample_rate,
# hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("MFCC coefficients")
# plt.colorbar()
# plt.title("MFCCs")
# plt.show()


# In[12]:


# audio_fpath = r'D:/03_AAU_Masters/03_3rd_semester/01_project/whale_data/data/sample/yes/'
#audio_fpath = "./train/"
# audio_fpath = "./test/"
# audio_clips = os.listdir(audio_fpath)
# FIG_SIZE = (8,6)

# def generate_spectrogram(signal, sample_rate, save_name):

#     hop_length = 128 # in num. of samples
#     n_fft = 2048 # window in num. of samples
#     stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

#     # calculate abs values on complex numbers to get magnitude
#     spectrogram = np.abs(stft)

#     # apply logarithm to cast amplitude to Decibels
#     log_spectrogram = librosa.amplitude_to_db(spectrogram)

#     # plotting the spectrogram
#     fig = plt.figure(figsize=FIG_SIZE, dpi=1000, frameon=False)
#     ax = fig.add_axes([0,0,1,1], frameon=False)
#     ax.axis('off')
#     librosa.display.specshow(log_spectrogram, sr=2000, hop_length=hop_length, cmap='gray', x_axis='time', y_axis='hz')
#     plt.savefig(save_name, pil_kwargs={'quality': 95}, bbox_inches=0, pad_inches=0)
#     librosa.cache.clear()

# # Creating sprectrograms for both train and test batch    
# for i in audio_clips:
# #     spectrograms_path = "./train_spectrograms/"
#     spectrograms_path = "./test_spectrograms/"
#     save_name = spectrograms_path + i + ".jpg" # i[:-5] without the .aiff
#     # check if a file already exists
#     if not os.path.exists(save_name):
#         signal, sample_rate = librosa.load(audio_fpath + i,sr = 2000) # your data sample rate is 2000 Hz, retrieved from the files
#         generate_spectrogram(signal, sample_rate, save_name)
#         plt.close()


# In[13]:


# Transforming images
datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)


# In[14]:


# Splitting the data for training and defining classes
train_generator=datagen.flow_from_dataframe(
dataframe=traindf[8500:],
directory='./train_spectrograms/',
x_col='clip_name',
y_col='label',
batch_size=10,
seed=42,
shuffle=True,
class_mode='categorical',
color_mode='grayscale',
classes=['0', '1'],
target_size=(129,500))


# In[15]:


# Splitting the data for validation and defining classes
valid_generator=datagen.flow_from_dataframe(
dataframe=traindf[-3000:],
directory='./train_spectrograms/',
x_col='clip_name',
y_col='label',
batch_size=10,
seed=42,
shuffle=True,
class_mode='categorical',
color_mode='grayscale',
classes=['0', '1'],
target_size=(129,500))


# In[16]:


# Splitting the data for testing
test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf[2000:],
directory='./test_spectrograms/',
x_col='clip_name',
y_col=None,    
batch_size=5,
seed=42,
shuffle=False,
class_mode=None,
color_mode='grayscale',
target_size=(129,500))


# In[17]:


# Initialising the model
model = Sequential()

# Adding the Convolutional layer
model.add(Conv2D(32, (3,3), input_shape = (129,500,1), activation=activations.relu))
# Adding the Max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Adding the Convolutional layer
model.add(Conv2D(32, (3,3), activation=activations.relu))
# Adding the Max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
# Implementing dropout
model.add(Dropout(0.2))

# Adding the Convolutional layer
model.add(Conv2D(64, (3,3), activation=activations.relu))
# Adding the Max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
# Implementing dropout
model.add(Dropout(0.2))
 
# Adding the Convolutional layer
model.add(Conv2D(64, (3,3), activation=activations.relu))
# Adding the Max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
# Implementing dropout
model.add(Dropout(0.2))

# Adding the Flattening data 
model.add(Flatten())

# Adding a fully conected layer
model.add(Dense(64, activation=activations.relu))
# Implementing dropout
model.add(Dropout(0.5))

# Adding a fully conected layer
model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

# Applying an optimizing algorithm
adam = keras.optimizers.Adam(lr=0.001)

# Running the model
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# Fitting the model
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
history = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10,
                    verbose=2
)


# In[ ]:


# Save the model to disk.
model.save_weights('whales-cnn1.h5')


# In[ ]:


# Resetting the generator
test_generator.reset()
pred=model.predict(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)


# In[ ]:


pred_bool = (pred > 0.5)


# In[ ]:


#acting like a test_generator, batch_size should be 1
model.evaluate(valid_generator, steps=STEP_SIZE_VALID) 

predictions=[]
labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
for row in pred_bool:
    l=[]
    for index,cls in enumerate(row):
        if cls:
            l.append(labels[index])
    predictions.append(",".join(l))
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)


# In[ ]:


df = pd.read_csv('results.csv')


# In[ ]:


print(df)


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Deep CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,31))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 31, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 31, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")


# In[ ]:




