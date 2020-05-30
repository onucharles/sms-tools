import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft, fftshift
import math
import matplotlib.pyplot as plt
eps = np.finfo(float).eps

""" 
A4-Part-1: Extracting the main lobe of the spectrum of a window

Write a function that extracts the main lobe of the magnitude spectrum of a window given a window 
type and its length (M). The function should return the samples corresponding to the main lobe in 
decibels (dB).

To compute the spectrum, take the FFT size (N) to be 8 times the window length (N = 8*M) (For this 
part, N need not be a power of 2). 

The input arguments to the function are the window type (window) and the length of the window (M). 
The function should return a numpy array containing the samples corresponding to the main lobe of 
the window. 

In the returned numpy array you should include the samples corresponding to both the local minimas
across the main lobe. 

The possible window types that you can expect as input are rectangular ('boxcar'), 'hamming' or
'blackmanharris'.

NOTE: You can approach this question in two ways: 1) You can write code to find the indices of the 
local minimas across the main lobe. 2) You can manually note down the indices of these local minimas 
by plotting and a visual inspection of the spectrum of the window. If done manually, the indices 
have to be obtained for each possible window types separately (as they differ across different 
window types).

Tip: log10(0) is not well defined, so its a common practice to add a small value such as eps = 1e-16 
to the magnitude spectrum before computing it in dB. This is optional and will not affect your answers. 
If you find it difficult to concatenate the two halves of the main lobe, you can first center the 
spectrum using fftshift() and then compute the indexes of the minimas around the main lobe.


Test case 1: If you run your code using window = 'blackmanharris' and M = 100, the output numpy 
array should contain 65 samples.

Test case 2: If you run your code using window = 'boxcar' and M = 120, the output numpy array 
should contain 17 samples.

Test case 3: If you run your code using window = 'hamming' and M = 256, the output numpy array 
should contain 33 samples.

"""
def extractMainLobe(window, M):
    """
    Input:
            window (string): Window type to be used (Either rectangular ('boxcar'), 'hamming' or '
                blackmanharris')
            M (integer): length of the window to be used
    Output:
            The function should return a numpy array containing the main lobe of the magnitude 
            spectrum of the window in decibels (dB).
    """

    w = get_window(window, M)         # get the window 
    
    ### Your code here

    #compute fft.
    N = 8 * M
    X = fft(w, n=N)
    X = fftshift(X)     # center fft around 0
    mX = 20 * np.log10(abs(X) + 1e-16)  # convert to db
    print("total no of samples:", len(mX))

    # calculate main lobe
    # find index of peak (ie peak of main lobe.)
    peak_idx = np.argmax(mX)

    # go through subsequent values until t+1 > t, pick t as minima.
    pos_minima = peak_idx
    for i in np.arange(peak_idx + 1, len(mX)):
        next_val = mX[i + 1]
        cur_val = mX[i]
        if next_val > cur_val:
            pos_minima = i
            print("Index of positive minima is: ", pos_minima)
            break
    
    # Get neg mimina. It is same distance from peak as pos minima.
    neg_minima = peak_idx - (pos_minima - peak_idx)
    print("Index of negative minima is: ", neg_minima)

    main_lobe = mX[neg_minima:pos_minima+1]
    print("length of main lobe: ", len(main_lobe))

    return main_lobe

ml = extractMainLobe("boxcar", 120)
# plot spectrum
plt.figure()
plt.plot(ml)
plt.show()