import librosa
import numpy as np
import recorder

class Recognizer:

    def __init__(self):
        recorder.Recorder().live()
        self.audio_data = "recording_live.wav"
        self.data, self.sampling_rate = librosa.load(self.audio_data)  #loading audio data
        print(type(self.data), type(self.sampling_rate))

    def recognize(self,model,labels):
        spectral_centroids = librosa.feature.spectral_centroid(y=self.data, sr=self.sampling_rate)[0]   #Extracting Spectral_centroids from audio
        rmse=librosa.feature.rms(y=self.data) #calculating rmse
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.data+0.01, sr=self.sampling_rate)[0]  #Extracting Sepectral rolloff
        spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y=self.data+0.01, sr=self.sampling_rate)[0]
        spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y=self.data+0.01, sr=self.sampling_rate, p=3)[0]
        spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y=self.data+0.01, sr=self.sampling_rate, p=4)[0]
        zero_crossings = librosa.zero_crossings(self.data, pad=False) #Calculating Zero Crossings
        mfccs = librosa.feature.mfcc(y=self.data, sr=self.sampling_rate) #Extracting Mel-frequency cepstral coeffecients (MFCCs)
        chroma = librosa.feature.chroma_stft(y=self.data, sr=self.sampling_rate)
        self.features=[np.mean(spectral_centroids),np.mean(rmse),np.mean(spectral_rolloff),np.mean(spectral_bandwidth_2),np.mean(spectral_bandwidth_3),np.mean(spectral_bandwidth_4),np.mean(zero_crossings),np.mean(mfccs),np.mean(chroma)]
        
        print(self.features)

        prediction=model.predict([self.features])
        sol=zip(prediction[0],labels)
        for x in sol: 
            print(x)
        #print(mfccs.shape)
        #print(mfccs)

    def analyze(self):
        print("analyzing")  
