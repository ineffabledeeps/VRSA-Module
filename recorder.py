# import required libraries
from asyncio.windows_events import NULL
from cProfile import label
import os
from time import sleep
import time
from tracemalloc import stop
from colorama import Fore
from sklearn import datasets
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import keyboard

class Recorder:
    
    #sync changes

    stop_recording=False
    label_name=NULL

    #Setting up parameters like freq, duration, samplerate, channels for recorder
    def __init__(self):
        
        self.freq = 44100     # Sampling frequency

        self.duration = 5    # Recording duration

        keyboard.add_hotkey('esc',lambda: self.stop_recording())  #creating Hotkey to stop program

        if not os.path.exists("dataset"):    # Creating folder dataset
            os.makedirs("dataset")

        if not os.path.exists("dataset/voice-samples"): #Creating folder datset/voice-samples
                os.makedirs("dataset/voice-samples")

    #-- END OF INITIALIZING CONSTRUCTOR --


    def record(self):
        
        counter=3 #Interval between recordings

        input("Recorder Ready! "+Fore.RED+"Press Enter "+Fore.WHITE) #Waiting for Enter key to start recording

        while(True):

            if(self.stop_recording==True):
                break

            # Start recorder with the given values
            # of duration and sample frequency
            self.recording = sd.rec(int(self.duration * self.freq),
    				samplerate=self.freq, channels=2)

            # Record audio for the given number of seconds
            print("Recording in progress...")
            sd.wait()

            
            if(not self.label_name):
                self.label_name=input('Enter label name: ')

            self.save_recording(self.label_name)
            time.sleep(1)
            for i in range(5): 
                if(i<1):
                    counter=3
                    break 
                print("Next recording session will start in: ",i)
                counter=counter-1
                time.sleep(1)

    #-- END OF RECORD --        

    def save_recording(self, name):
        counter=0
        recording_path="dataset/voice-samples/"+str(name) #Path to store labeled recordings

        while True:
            if os.path.exists(recording_path+"/recording"+str(counter)+".wav"):
                counter=counter+1
            else:
                break

        
        if not os.path.exists(recording_path):
            print("creating Label directory")
            os.makedirs(recording_path)

        # This will convert the NumPy array to an audio
        # file with the given sampling frequency        
        #write(recording_path+"/recording"+str(counter)+".wav", self.freq, self.recording)
        
        # Convert the NumPy array to audio file with sample width
        wv.write(recording_path+"/recording"+str(counter)+".wav", self.recording, self.freq, sampwidth=2)
        
        print(recording_path+"/recording"+str(counter)+".wav "+Fore.GREEN+"Saved"+Fore.WHITE)

    #-- END OF SAVE_RECORDING --

    def stop_recording(self):
        self.stop_recording=True
        print("Quitting in progress")

    #-- END OF STOP_RECORDING --   

x = Recorder()
x.record()