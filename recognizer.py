import librosa
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import recorder


class Recognizer:

    def __init__(self):
        recorder.Recorder().live()
        self.audio_data = "recording_live.wav"
        self.data, self.sampling_rate = librosa.load(self.audio_data)  #loading audio data
        print(type(self.data), type(self.sampling_rate))

    def recognize(self,model,labels):
        self.spectral_centroids = librosa.feature.spectral_centroid(y=self.data, sr=self.sampling_rate)[0]   #Extracting Spectral_centroids from audio
        self.rmse=librosa.feature.rms(y=self.data) #calculating rmse
        self.spectral_rolloff = librosa.feature.spectral_rolloff(y=self.data+0.01, sr=self.sampling_rate)[0]  #Extracting Sepectral rolloff
        self.spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y=self.data+0.01, sr=self.sampling_rate)[0]
        self.spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y=self.data+0.01, sr=self.sampling_rate, p=3)[0]
        self.spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y=self.data+0.01, sr=self.sampling_rate, p=4)[0]
        self.zero_crossings = librosa.zero_crossings(self.data, pad=False) #Calculating Zero Crossings
        self.mfccs = librosa.feature.mfcc(y=self.data, sr=self.sampling_rate) #Extracting Mel-frequency cepstral coeffecients (MFCCs)
        self.chroma = librosa.feature.chroma_stft(y=self.data, sr=self.sampling_rate)
        self.features=[np.mean(self.spectral_centroids),np.mean(self.rmse),np.mean(self.spectral_rolloff),np.mean(self.spectral_bandwidth_2),np.mean(self.spectral_bandwidth_3),np.mean(self.spectral_bandwidth_4),np.mean(self.zero_crossings)]

        for x in self.mfccs:
            self.features.append(np.mean(x))

        for y in self.chroma:
            self.features.append(np.mean(y))
            
        #print(self.mfccs)
        #print(self.chroma)
        print(len(self.features))

        prediction=model.predict([self.features])
        sol=zip(prediction[0],labels)
        for x in sol: 
            print(x)
        #print(mfccs.shape)
        #print(mfccs)


    def analyze(self):

        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        print("analyzing")  

        
        
        #Computing the time variable for visualization
        fig1, ax1=plt.subplots(nrows=3,sharex=True)
        frames = range(len(self.spectral_centroids))
        t = librosa.frames_to_time(frames)

        #Plotting the Spectral Centroid along the waveform
        librosa.display.waveshow(y=self.data, sr=self.sampling_rate, ax=ax1[0])  
        ax1[0].plot(t, normalize(self.spectral_centroids), color='b')

        librosa.display.waveshow(y=self.data, sr=self.sampling_rate, ax=ax1[1])
        ax1[1].plot(t, normalize(self.spectral_rolloff), color='r')

        fig2, ax2 = plt.subplots(nrows=3,sharex=True)
        librosa.display.waveshow(self.data, sr=self.sampling_rate,ax=ax1[2])
        ax1[2].plot(t, normalize(self.spectral_bandwidth_2), color='r')
        ax1[2].plot(t, normalize(self.spectral_bandwidth_3), color='g')
        ax1[2].plot(t, normalize(self.spectral_bandwidth_4), color='y')
        ax1[2].legend(('p = 2', 'p = 3', 'p = 4'))

        fig2, ax2=plt.subplots(nrows=2,sharex=True)

        print(sum(self.zero_crossings))



        fig3, ax3 = plt.subplots(nrows=2, sharex=True)
        img = librosa.display.specshow(librosa.power_to_db(S = np.abs(librosa.stft(self.data)), ref=np.max),x_axis='time', y_axis='mel', fmax=8000,ax=ax3[0])
        fig3.colorbar(img, ax=[ax3[0]])
        ax3[0].set(title='Mel spectrogram')
        ax3[0].label_outer()
        img = librosa.display.specshow(self.mfccs, x_axis='time', ax=ax3[1])
        fig3.colorbar(img, ax=[ax3[1]])
        ax3[1].set(title='MFCC')


        print(self.chroma.shape)
        #print(label+"-----"+rec)

        librosa.display.specshow(librosa.amplitude_to_db(S = np.abs(librosa.stft(self.data)), ref=np.max),y_axis='log', x_axis='time')
        librosa.display.specshow(self.chroma, y_axis='chroma', x_axis='time')
        plt.show()

#r=Recognizer()
#r.recognize()
        