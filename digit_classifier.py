import librosa
import librosa.display
import glob
import random
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling3D
import numpy as np
import matplotlib.pyplot as plt


audio_file_for_analysis = []
dataset = []
count = 0
count_2_jackson = 0
count_2_nicolas = 0
count_2_theo = 0
count_1_jackson = 0
count_1_nicolas = 0
count_1_theo = 0
count_1 = 0

#load all the audio files from the directory into a list
list_of_audio_files = [f for f in glob.glob('/Users/User/Downloads/free-spoken-digit-dataset-master/recordings/*.wav')]

for audio_file in list_of_audio_files:
    count += 1

    #set a threshoold to know how many audio files has been processed
    if count%100 == 0:
        print('done loading',count)

    #loading the audio file as time series and extracting its sampling rate
    data,sampling_rate = librosa.load(audio_file)

    #extract mfcc features form the extracted time series
    feature = librosa.feature.mfcc(y = data,sr = sampling_rate)

    #reshaoe the 2D matrix to 20x30
    padded_feature = librosa.util.fix_length(feature,30)

    #get the digits . In this case 2 as 1 and 1 ans 0
    fields = audio_file.split('/')
    temp = fields[len(fields)-1]
    name = temp.split('_')

    #If the digit is 2 set that as 1
    if name[0] is '2':
        if name[1] == 'jackson':
            count_2_jackson +=1
        if count_2_jackson == 1:
            audio_file_for_analysis.append((audio_file,2))
        if name[1] == 'nicolas':
            count_2_nicolas +=1
        if count_2_nicolas ==1:
            audio_file_for_analysis.append((audio_file, 2))
        if name[1] == 'theo':
            count_2_theo +=1
        if count_2_theo == 1:
            audio_file_for_analysis.append((audio_file,2))
        target = '1'
        dataset.append((padded_feature, target))

    #if the digit is  1 set that as 0
    if name[0] is '1':
        if name[1] == 'jackson':
            count_1_jackson +=1
        if count_1_jackson == 1:
            audio_file_for_analysis.append((audio_file,1))
        if name[1] == 'nicolas':
            count_1_nicolas +=1
        if count_1_nicolas ==1:
            audio_file_for_analysis.append((audio_file, 1))
        if name[1] == 'theo':
            count_1_theo +=1
        if count_1_theo == 1:
            audio_file_for_analysis.append((audio_file,1))
        target = '0'
        dataset.append((padded_feature,target))

#shuffle the dataset at random
random.shuffle(dataset)

#split the dataset into training and testing set with 80% of the data into training set
train = dataset[:240]
test = dataset[240:]

#split the features (sudio files and the target values (digit)
x_train,y_train = zip(*train)
x_test,y_test = zip(*test)

#reshapoe the audio file such that it has a 3D shape.Also, the audio time series of eeach audio varies. Hence we pad it into 20x30
x_train = np.array([x.reshape( (20, 30) ) for x in x_train])
x_test = np.array([x.reshape( (20, 30) ) for x in x_test])

#Apply onehot encodng to categorical variables
y_train = np.array(keras.utils.to_categorical(y_train,num_classes=None))
y_test = np.array(keras.utils.to_categorical(y_test,num_classes=None))


#build a sequential model
model = Sequential()
#add the layer of neurons
model.add(LSTM(128,activation='sigmoid'))
model.add(Dense(300,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='softmax'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))

#compile the model
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#train the model using the training set
model.fit(x_train,y_train,epochs=10,batch_size=50)

#evaluate the model using the testing set
acc = model.evaluate(x_test, y_test,batch_size=50)

print('Loss and accuracy values are as follows:', acc)

#predict the target variables in the testing set
predicted = model.predict(x_test,verbose=0)

#uncomment it if you want to print the output
'''
for p in predicted:
    if p[0] > p[1]:
        print ('0')
    else:
        print('1')
'''

#print the model summary
print(model.summary())


#analysis


def graphs(data,name,):
    #data, sampling_rate = librosa.load(file_name)
    feature = librosa.feature.mfcc(y=data, sr=sampling_rate)
    #print the sampling rate of the audio file
    print('sampling rate for digit', digit, ':', sampling_rate)

    # wave plot for the audio file
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(data, sr=sampling_rate)
    plt.title((name[0], name[1], 'waveplot'))
    plt.show()

    # spectrogram for the audio file - Spectral features
    librosa.display.specshow(feature)
    plt.xlabel('time')
    plt.ylabel('hz')
    plt.title((name[0], name[1], 'Spectrogram'))
    plt.show()

    # power spectrogram based on the mfcc frequency types for the audio file
    mfcc_spectral_analysis = librosa.amplitude_to_db(np.abs(librosa.feature.mfcc(data, sr=sampling_rate)), ref=np.max)
    librosa.display.specshow(mfcc_spectral_analysis, y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title((name[0], name[1], 'MFCC power spectrum'))
    plt.show()
'''
for file in audio_file_for_analysis:
    tuple_to_list = list(file)
    file_name = tuple_to_list[0]
    digit = tuple_to_list[1]
    fields = file_name.split('/')
    temp = fields[len(fields) - 1]
    name = temp.split('_')
    data,sampling_rate = librosa.load(file_name)
    graphs(data,name)
'''
#add noise:

def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise

print('adding noise')
tuple_to_list = list(audio_file_for_analysis[0])
file_name = tuple_to_list[0]
digit = tuple_to_list[1]
fields = file_name.split('/')
temp = fields[len(fields) - 1]
name = temp.split('_')

data,sampling_rate = librosa.load(file_name)
noise_data = add_noise(data)
#plot after adding the noise
graphs(noise_data,name)

#filters noise
print('removing noise')
reduced_noise = librosa.decompose.nn_filter(noise_data,aggregate=np.median,metric='cosine')
#plot after removine the noise
graphs(reduced_noise,['1','jackson'])


