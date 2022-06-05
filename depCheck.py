import sys
import subprocess
import time
import pkg_resources

print("Running dependency check..")
required = {'opencv-python', 'tk','imutils','pillow'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print("Oops! looks like you are missing some libraries \nSit back and relax we are taking care of everything.")
    python = sys.executable
    returned_value = subprocess.call([python, '-m', 'pip', 'install', *missing], shell=True, universal_newlines=True)
    print('returned value:', returned_value)
print("We are good to go!")