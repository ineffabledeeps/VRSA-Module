#import IPython.display as ipd
#ipd.Audio('recording1.wav')

import os
import pandas as pd
import librosa
import librosa.display
import glob 
import matplotlib.pyplot as plt

data, sampling_rate = librosa.load('recording1.wav')


plt.figure(figsize=(12, 4))
fig, ax = plt.subplots(nrows=1, sharex=True)
librosa.display.waveshow(data, sr=sampling_rate, ax=ax)
ax.set(title='Envelope view, mono')
ax.label_outer()

#print(train)
plt.show()