import sys
import time
import depCheck
from colorama import Fore

import recorder
import trainer
import recognizer

class main:

    def __init__(self):
        margin=15
        print("\n")
        print(" "*margin+"*"*71+"*")
        print(" "*margin+"*"+" "*70+"*")
        print(" "*margin+"*"+2*" "+Fore.GREEN+"Welcome to Voice Recognition, Synthesis and Analyzer Module (VRSA)"+Fore.WHITE+2*" "+"*")
        print(" "*margin+"*"+" "*70+"*")
        print(" "*margin+"*"*71+"*")
        print("\n")

        time.sleep(3)
        def vrsa_help():
            print(Fore.GREEN+"\nAvailable Commands")
            print(18*"-")
            print("-record")
            print("-recognize")
            print("-analyze")
            print("-synt")
            print("-exit")
            print("-help")
            print("-info\n"+Fore.WHITE)

        vrsa_help()

        def vrsa_info():
            print("Project: Voice Recognition, Synthesis and Analyzer Module (VRSA) Module")
            print("Submitted to: DYPIU")
            print("Contributers: Deepak Bobade, Mayank Nipane, Sudhanshu Jichkar, Juee Jadhav")

        while True:
            input_choice=input("Command: ")

            if(input_choice=='exit'):
                break

            elif(input_choice=='info'):
                vrsa_info()

            elif(input_choice=='help'):
                vrsa_help()

            elif(input_choice=='record'):
                self.rercorder_instance=recorder.Recorder()
                self.rercorder_instance.record()

            elif(input_choice=='recognize'):
                self.trainer_instance=trainer.Trainer()
                #self.trainer_instance.extract()
                model=self.trainer_instance.train()

                self.recognizer_instance=recognizer.Recognizer()
                self.recognizer_instance.recognize(model)

            else:
                print("Invalid command try from below commands")
                vrsa_help()

        
main()