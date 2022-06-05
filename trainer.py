import csv
import os
from tkinter import W
from typing import Counter
import pandas as pd
import sklearn
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
class Trainer:

    def __init__(self):

        #Creating Fields for CSV file
        self.head=["id","name","spectral_centroid","rmse","spectral_rolloff","spectral_bandwidth_2","spectral_bandwidth_3","spectral_bandwidth_4","zero_crossings","mfccs","chroma"]

        #checking if training_info.csv exists or not
        #if not exists will create csv file
        if(not os.path.exists("training_info.csv")):
            with open("training_info.csv",'w',newline="") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(self.head)        

    # -- END OF INTITALIZING CONSTRUCTOR --

    def extract(self): 

        labels=[x for x in os.listdir(f"dataset/voice-samples")]
        print(labels)
        for label in labels:
            for rec in [x for x in os.listdir(f"dataset/voice-samples/{label}")]:
                audio_data = f'dataset/voice-samples/{label}/{rec}'
                data, sampling_rate = librosa.load(audio_data)  #loading audio data
                print(type(data), type(sampling_rate))

                spectral_centroids = librosa.feature.spectral_centroid(y=data, sr=sampling_rate)[0]   #Extracting Spectral_centroids from audio
                print(spectral_centroids.shape)

                # Computing the time variable for visualization
                #fig1, ax1=plt.subplots(nrows=3,sharex=True)
                frames = range(len(spectral_centroids))
                t = librosa.frames_to_time(frames)

                # Normalising the spectral centroid for visualisation
                def normalize(x, axis=0):
                    return sklearn.preprocessing.minmax_scale(x, axis=axis)

                #Plotting the Spectral Centroid along the waveform
                #librosa.display.waveshow(y=data, sr=sampling_rate, ax=ax1[0])  
                #ax1[0].plot(t, normalize(spectral_centroids), color='b')

                rmse=librosa.feature.rms(y=data) #calculating rmse

                spectral_rolloff = librosa.feature.spectral_rolloff(y=data+0.01, sr=sampling_rate)[0]  #Extracting Sepectral rolloff
                #librosa.display.waveshow(y=data, sr=sampling_rate, ax=ax1[1])
                #ax1[1].plot(t, normalize(spectral_rolloff), color='r')

                spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y=data+0.01, sr=sampling_rate)[0]
                spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y=data+0.01, sr=sampling_rate, p=3)[0]
                spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y=data+0.01, sr=sampling_rate, p=4)[0]

                #fig2, ax2 = plt.subplots(nrows=3,sharex=True)
                #librosa.display.waveshow(data, sr=sampling_rate,ax=ax1[2])
                #ax1[2].plot(t, normalize(spectral_bandwidth_2), color='r')
                #ax1[2].plot(t, normalize(spectral_bandwidth_3), color='g')
                #ax1[2].plot(t, normalize(spectral_bandwidth_4), color='y')
                #ax1[2].legend(('p = 2', 'p = 3', 'p = 4'))

                #fig2, ax2=plt.subplots(nrows=2,sharex=True)

                zero_crossings = librosa.zero_crossings(data, pad=False) #Calculating Zero Crossings
                print(sum(zero_crossings))

                mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate) #Extracting Mel-frequency cepstral coeffecients (MFCCs)
                print(mfccs.shape)

                #fig3, ax3 = plt.subplots(nrows=2, sharex=True)
                #img = librosa.display.specshow(librosa.power_to_db(S = np.abs(librosa.stft(data)), ref=np.max),x_axis='time', y_axis='mel', fmax=8000,ax=ax3[0])
                #fig3.colorbar(img, ax=[ax3[0]])
                #ax3[0].set(title='Mel spectrogram')
                #ax3[0].label_outer()
                #img = librosa.display.specshow(mfccs, x_axis='time', ax=ax3[1])
                #fig3.colorbar(img, ax=[ax3[1]])
                #ax3[1].set(title='MFCC')

                chroma = librosa.feature.chroma_stft(y=data, sr=sampling_rate)
                print(chroma.shape)
                print(label+"-----"+rec)

                #librosa.display.specshow(librosa.amplitude_to_db(S = np.abs(librosa.stft(data)), ref=np.max),y_axis='log', x_axis='time')
                #librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
                #plt.show()

                #Generating feature id (can be found on first column of csv)
                def generateId():
                    counter=0
                    with open("training_info.csv",'r',newline='') as csvfile:
                        csvreader=csv.reader(csvfile)
                        lines=[line for line in csvreader]    
                        for line in lines[1:]:
                            if int(line[0])==counter:
                                counter=counter+1
                    return counter

                #Writing data to csv file
                with open("training_info.csv",'a',newline='') as csvfile:
                    csvwriter=csv.writer(csvfile)
                    csvwriter.writerow([generateId(),label,np.mean(spectral_centroids),np.mean(rmse),np.mean(spectral_rolloff),np.mean(spectral_bandwidth_2),np.mean(spectral_bandwidth_3),np.mean(spectral_bandwidth_4),np.mean(zero_crossings),np.mean(mfccs),np.mean(chroma)])
             

        #-- END OF EXTRACT --    

x=Trainer()
x.extract()