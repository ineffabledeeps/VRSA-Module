import sklearn
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_data = 'dataset/voice-samples/Deepak/recording0.wav'
data, sampling_rate = librosa.load(audio_data)  #loading audio data
print(type(data), type(sampling_rate))

spectral_centroids = librosa.feature.spectral_centroid(y=data, sr=sampling_rate)[0]   #Extracting Spectral_centroids from audio
print(spectral_centroids.shape)

# Computing the time variable for visualization
fig1, ax1 = plt.subplots(nrows=3,sharex=True)
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#Plotting the Spectral Centroid along the waveform
librosa.display.waveshow(y=data, sr=sampling_rate, ax=ax1[0])
ax1[0].plot(t, normalize(spectral_centroids), color='b')

spectral_rolloff = librosa.feature.spectral_rolloff(y=data+0.01, sr=sampling_rate)[0]
librosa.display.waveshow(y=data, sr=sampling_rate, ax=ax1[1])
ax1[1].plot(t, normalize(spectral_rolloff), color='r')

spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y=data+0.01, sr=sampling_rate)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y=data+0.01, sr=sampling_rate, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y=data+0.01, sr=sampling_rate, p=4)[0]

#fig2, ax2 = plt.subplots(nrows=3,sharex=True)
librosa.display.waveshow(data, sr=sampling_rate,ax=ax1[2])
ax1[2].plot(t, normalize(spectral_bandwidth_2), color='r')
ax1[2].plot(t, normalize(spectral_bandwidth_3), color='g')
ax1[2].plot(t, normalize(spectral_bandwidth_4), color='y')
ax1[2].legend(('p = 2', 'p = 3', 'p = 4'))

plt.show()
