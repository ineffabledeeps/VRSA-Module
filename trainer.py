import sklearn
import librosa
import matplotlib.pyplot as plt

audio_data = 'dataset/voice-samples/Deepak/recording0.wav'
data, sampling_rate = librosa.load(audio_data)  #loading audio data
print(type(data), type(sampling_rate))

spectral_centroids = librosa.feature.spectral_centroid(data, sr=sampling_rate)[0]   #Extracting Spectral_centroids from audio
print(spectral_centroids.shape)

# Computing the time variable for visualization
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')