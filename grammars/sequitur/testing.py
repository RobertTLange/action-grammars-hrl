import os
import subprocess

command = 'echo' + str("abracadra")+ '| ./sequitur -p -k 2'

with open(os.devnull, 'wb') as devnull:
        subprocess.check_call([command], stdout=devnull, stderr=subprocess.STDOUT)
