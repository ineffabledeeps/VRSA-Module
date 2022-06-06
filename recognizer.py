import librosa

class recognizer:
    def recognize():
        def __init__():
        audio_data = f'dataset/voice-samples/{label}/{rec}'
        data, sampling_rate = librosa.load(audio_data)  #loading audio data
        print(type(data), type(sampling_rate))


        spectral_centroids = librosa.feature.spectral_centroid(y=data, sr=sampling_rate)[0]   #Extracting Spectral_centroids from audio
        rmse=librosa.feature.rms(y=data) #calculating rmse
        spectral_rolloff = librosa.feature.spectral_rolloff(y=data+0.01, sr=sampling_rate)[0]  #Extracting Sepectral rolloff
        spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y=data+0.01, sr=sampling_rate)[0]
        spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y=data+0.01, sr=sampling_rate, p=3)[0]
        spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y=data+0.01, sr=sampling_rate, p=4)[0]
        zero_crossings = librosa.zero_crossings(data, pad=False) #Calculating Zero Crossings
        mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate) #Extracting Mel-frequency cepstral coeffecients (MFCCs)
        chroma = librosa.feature.chroma_stft(y=data, sr=sampling_rate)