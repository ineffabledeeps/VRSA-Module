import sys
import subprocess
import time
import pkg_resources

#sklearn pandas tensorflow librosa matplotlib numpy sounddevice sci ,'wav ,'keyboard colorama keras"
print("Running dependency check..")
required = {'sklearn','pandas','tensorflow','librosa','matplotlib','numpy','sounddevice','scipy','wavio','keyboard','colorama','keras'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print("Oops! looks like you are missing some libraries \nSit back and relax we are taking care of everything.")
    python = sys.executable
    returned_value = subprocess.call([python, '-m', 'pip', 'install', *missing], shell=True, universal_newlines=True)
    print('returned value:', returned_value)
print("We are good to go!")