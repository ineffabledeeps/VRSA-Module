import sys
import time

from colorama import Fore

margin=15
print("\n")
print(" "*margin+"*"*71+"*")
print(" "*margin+"*"+" "*70+"*")
print(" "*margin+"*"+2*" "+Fore.GREEN+"Welcome to Voice Recognition, Synthesis and Analyzer Module (VRSA)"+Fore.WHITE+2*" "+"*")
print(" "*margin+"*"+" "*70+"*")
print(" "*margin+"*"*71+"*")
print("\n")

import depCheck
time.sleep(3)
print(Fore.GREEN+"\nAvailable Commands")
print(18*"-")
print("-record")
print("-analyze")
print("-synt")
print("-exit")
print("-info\n"+Fore.WHITE)

while True:
    input_choice=input("Command: ")

    if(input_choice=='exit'):
        break


