import sys
import os
sys.path.append('../../software/models/')
from utilFunctions import wavread
from pathlib import Path

"""
A1-Part-2: Basic operations with audio

Write a function that reads an audio file and returns the minimum and the maximum values of the audio 
samples in that file. 

The input to the function is the wav file name (including the path) and the output should be two floating 
point values returned as a tuple.

If you run your code using oboe-A4.wav as the input, the function should return the following output:  
(-0.83486432, 0.56501967)
"""
def minMaxAudio(inputFile):
    """
    Input:
        inputFile: file name of the wav file (including path)
    Output:
        A tuple of the minimum and the maximum value of the audio samples, like: (min_val, max_val)
    """
    ## Your code here
    wav_file = Path(inputFile)
    if not wav_file.is_file():
        raise ValueError(inputFile + " file does not exist")
    _, x = wavread(inputFile)
    min_x, max_x = min(x), max(x)
    return (min_x, max_x)

print(minMaxAudio('../../sounds/oboe-A4.wav'))
