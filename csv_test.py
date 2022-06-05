import csv
import os

#Creating Fields for CSV file
head=["id","name","spectral_centroid","spectral_rolloff","spectral_bandwidth_2","spectral_bandwidth_3","spectral_bandwidth_4","zero_crossings","mfccs","chroma"]

#checking if training_info.csv exists or not
#if not exists will create csv file
if(not os.path.exists("training_info.csv")):
    with open("training_info.csv",'w',newline="") as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(head)   