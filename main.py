import sys
import time
import depCheck

from colorama import Fore

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
    print(Fore.GREEN+"Available Commands")
    print(18*"-")
    print("-record")
    print("-analyze")
    print("-synt")
    print("-exit")
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

    if(input_choice=='info'):
        vrsa_info()

    if(input_choice=='help'):
        vrsa_help()