import os
import threading
import time
import numpy as np

from threading import Thread
from threading import Event

event = Event()

# Define a function for the thread
def get_gpu(event,delay):
    
   while (1):
       if event.is_set():
            break
       time.sleep(delay)
       os.system("nvidia-smi | awk '$2==\"N/A\"{print substr($9,1,length($9)-3),substr($11,1,length($11)-3),substr($13,1,length($13)-1)}' > smi_tmp.txt")
       
# Create two threads as follows
try:
    x = Thread(target=print_time, args=(event,1,))
    x.start()
except:
   print("Error: unable to start thread")


for i in range(3):
    time.sleep(3.5)
    print(np.loadtxt('smi_tmp.txt'))

event.set()

x.join()
print('everything done')

