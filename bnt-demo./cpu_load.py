import os
import signal
import time
import sys

from multiprocessing import Pool
from multiprocessing import cpu_count

def handler(signum, stack):
    #print ('Alarm: ')
    sys.exit()

def f(x):
    while True:
        x*x
        print("CPU number :", x)

signal.signal(signal.SIGALRM, handler)
signal.alarm(60)
#time.sleep(10)

processes = cpu_count()
print('-' * 20)
print('Running load on CPU(s)')
print('Utilizing %d cores' % processes)
print('-' * 20)
pool = Pool(processes)
pool.map(f, range(processes))
