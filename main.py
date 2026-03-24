

import os
import time
import argparse


import time
import tensorflow as tf
import tensorflow_io as tfio


def print_test():
    # do math multiplications and then print test

    for i in range(10):
        a = i * 2
        b = i * 3
        c = a + b
        print(f"Test {i}: {c}")

    

def main():

    print_test()

    time.sleep(600)

if __name__ == "__main__":
    main()

