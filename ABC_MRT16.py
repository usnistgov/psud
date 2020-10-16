import numpy as np
import numpy.matlib
import scipy.io as sio
import math
from warnings import warn

# --------------------------Background--------------------------
# ABC_MRT16.m implements the ABC-MRT16 algorithm for objective estimation of 
# speech intelligibility.  The algorithm is discussed in detail in [1] and
# [2].
# 
# The Modified Rhyme Test (MRT) is a protocol for evaluating speech
# intelligibility using human subjects [3]. The subjects are presented
# with the task of identifying one of six different words that take the
# phonetic form CVC.  The six options differ only in the leading or
# trailing consonant. MRT results take the form of success rates
# (corrected for guessing) that range from 0 (guessing) to 1 
# (correct identification in every case).  These success rates form a 
# measure of speech intelligibility in this specific (MRT) context.
# 
# The 2016 version of Articulation Band Correlation-MRT, (ABC-MRT16) is a 
# signal processing algorithm that processes MRT audio files and produces
# success rates.
# 
# The goal of ABC-MRT16 is to produce success rates that agree with those
# produced by MRT. Thus ABC-MRT16 is an automated or objective version of
# MRT and no human subjects are required. ABC-MRT16 uses a very simple and
# specialized speech recognition algorithm to decide which word was spoken.
# This version has been tested on narrowband, wideband, superwideband,
# and fullband speech.
# 
# Information on preparing test files and running ABC_MRT16.m can be found
# in the readme file included in the distribution.  ABC_MRTdemo16.m shows
# example use.
# 
# --------------------------References--------------------------
# [1] S. Voran "Using articulation index band correlations to objectively
# estimate speech intelligibility consistent with the modified rhyme test,"
# Proc. 2013 IEEE Workshop on Applications of Signal Processing to Audio and
# Acoustics, New Paltz, NY, October 20-23, 2013.  Available at
# www.its.bldrdoc.gov/audio.
# 
# [2] S. Voran " A multiple bandwidth objective speech intelligibility 
# estimator based on articulation index band correlations and attention,"
# Proc. 2017 IEEE International Conference on Acoustics, Speech, and 
# Signal Processing, New Orleans, March 5-9, 2017.  Available at
# www.its.bldrdoc.gov/audio.
# 
# [3] ANSI S3.2, "American national standard method for measuring the 
# intelligibility of speech over communication systems," 1989.
# 
# --------------------------Legal--------------------------
# THE NATIONAL TELECOMMUNICATIONS AND INFORMATION ADMINISTRATION,
# INSTITUTE FOR TELECOMMUNICATION SCIENCES ("NTIA/ITS") DOES NOT MAKE
# ANY WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR STATUTORY, INCLUDING,
# WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR
# A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY.  THIS SOFTWARE
# IS PROVIDED "AS IS."  NTIA/ITS does not warrant or make any
# representations regarding the use of the software or the results thereof,
# including but not limited to the correctness, accuracy, reliability or
# usefulness of the software or the results.
# 
# You can use, copy, modify, and redistribute the NTIA/ITS developed
# software upon your acceptance of these terms and conditions and upon
# your express agreement to provide appropriate acknowledgments of
# NTIA's ownership of and development of the software by keeping this
# exact text present in any copied or derivative works.
# 
# The user of this Software ("Collaborator") agrees to hold the U.S.
# Government harmless and indemnifies the U.S. Government for all
# liabilities, demands, damages, expenses, and losses arising out of
# the use by the Collaborator, or any party acting on its behalf, of
# NTIA/ITS' Software, or out of any use, sale, or other disposition by
# the Collaborator, or others acting on its behalf, of products made
# by the use of NTIA/ITS' Software.
 
class ABC_MRT16:
#This class creates a new ABC_MRT16 object 
#This loads in the speech templates for the MRT words 
#The object can then be used to calculate intelligibility for audio clips
    
    def parse_templates():
    #ABC_MRT_FB_templates.mat contains 1 by 1200 cell array TFtemplatesFB  
    #Each cell contains a fullband time-frequency template for one of the 1200 talker x keyword combinations
    
        ABC_MRT_FB_templates_contents = sio.loadmat('ABC_MRT_FB_templates.mat')
        TFtemplatesFB = ABC_MRT_FB_templates_contents['TFtemplatesFB']
        return TFtemplatesFB

    def makeAI():
    #This function makes the 21 by 215 matrix that maps FFT bins 1 to 215 to 21
    #AI bands. These are the AI bands specified on page 38 of the book: 
    #S. Quackenbush, T. Barnwell and M. Clements, "Objective measures of
    #speech quality," Prentice Hall, Englewood Cliffs, NJ, 1988.
        AIlims = np.array([[4,4], #AI band 1
                       [5,6],
                       [7,7],
                       [8,9],
                       [10,11],
                       [12,13],
                       [14,15],
                       [16,17],
                       [18,19],
                       [20,21],
                       [22,23],
                       [24,26],
                       [27,28],
                       [29,31],
                       [32,35],
                       [36,40], #AI band  16    
                       [41,45], #AI band 17
                       [46,52], #AI band  18
                       [53,62], #AI band  19
                       [63,76], #AI band  20
                       [77,215]]) #Everything above AI band 20 and below 20 kHz makes "AI band 21"
    
        AI = np.zeros((21, 215))
        for k in range(21):
            firstfreq = AIlims[k, 0]
            lastfreq = AIlims[k, 1]
            AI[k, (firstfreq - 1):lastfreq] = 1
    
        return AI
    
    ALIGN_BINS = np.arange(6,9) #FFT bins to use for time alignment   
    AI = makeAI() #Make 21 by 215 matrix that maps 215 FFT bins to 21 AI bands 
    templates = parse_templates()  #Templates for data
    binsPerBand = np.sum(AI, axis = 1, keepdims = True)  #Number of FFT bins in each AI band

    def process(obj, speech, file_num, verbose = False):
    #process - calculate of speech intelligibility with ABC-MRT16
    #
    #This function calculates the intelligibility of the speech given in list of numpy arrays speech 
    #file_num gives the word number and should have the same number of elements as speech 
    #the average intelligibility, corrected for guessing, over all words is given by phi_hat
    #the intelligibility of each individual word, not corrected for guessing, is returned in success. 
    
        success = np.zeros(len(speech))
        
        #padding speech to minimum length 
        speech = [padSpeech(s) for s in speech]
           
        for k in range(len(speech)):
            
            #calculate autocorrelation for speech
            #xcm = np.min(np.correlate(speech[k], speech[k], mode = "full")/np.sqrt(np.inner(speech[k], speech[k]) * np.inner(speech[k], speech[k])))
            
            #check for empty speech vector
            if np.size(speech[k]) == 0 or math.isnan(file_num[k]):
                success[k] = np.nan
                #check for speech using autocorrelation
                #if the signals are periodic, there will be anticorrelation
                #if the signals are noise, there will be no anticorrelation
                #NaN is returned from xcorr if the autocorrelation at lag zero is 0 because of normalization
           
            else:
                #calculate autocorrelation for speech
                xcm = np.min(np.correlate(speech[k], speech[k], mode = "full")/np.sqrt(np.inner(speech[k], speech[k]) * np.inner(speech[k], speech[k])))
                
                if xcm > -0.1 or math.isnan(xcm):
                    #speech not detected, skip algorithm
                    success[k] = 0
                
                    if verbose == True:
                        msg = f'In clip #{k}, speech not detected'
                        warn(msg)    
                else:
                    if verbose == True:
                        msg = f'Working on clip {k} of {len(speech)}'
                        print(msg, '\n')
                    
                    C = np.zeros((215,6))
           
                    #create time-freq representation and apply Stevens' Law
                    X = np.abs(T_to_TF(speech[k])) ** 0.6
           
                    #pointer that indicates which of the 6 words in the list was spoken in the .wav file 
                    #this is known in advance from file_num
                    #as file_num runs from 1 to 1200, correct word runs from 1 to 6, 200 times
                    correct_word = (file_num[k] - 1) % 6 
                
                    #pointer to first of the six words in the list associated with the present speech file 
                    #as file_num runs from 1 to 1200, first_word is 1 1 1 1 1 1 7 7 7 7 7 7 ...1195 1195 1195 1195 1195 1195
                    first_word = 6 * (math.floor((file_num[k] - 1) / 6) + 1) - 5
                
                    #compare the computed TF representation for the input .wav file with the TF templates for the 6 candidate words
                    for word in range(6):
                        #find number of columns (time samples) in template 
                        ncols = obj.templates[0, (first_word - 1 + word)].shape[1]
                    
                        #do correlation using a group of rows to find best time alignment between X and template    
                        shift = group_corr(X[obj.ALIGN_BINS, :], obj.templates[0, (first_word - 1 + word)][obj.ALIGN_BINS, :])
                    
                        #extract and normalize the best-aligned portion of X
                        temp = X[:, (shift+1):(shift + ncols+1)]
                    
                        XX = TFnorm(temp)
                    
                        #find correlation between XX and template, one result per FFT bin
                        C[:, word] = np.sum(np.multiply(XX, obj.templates[0, (first_word - 1 + word)]), axis = 1)

                    binsPerBand_tiled = obj.binsPerBand
                    binsPerBand_tiled = np.matlib.repmat(binsPerBand_tiled, 1, 6)
                
                    C = np.true_divide((obj.AI @ C), binsPerBand_tiled) #aggregate correlation values across each AI band
                    C = np.maximum(C, 0) #clamp
                
                    C = np.sort(C, axis = 0)
                    SAC = np.flip(C, axis = 0)
                                
                    #for each of the 6 word options, sort the 21 AI band correlations from largest to smallest
                    SAC = SAC[0:16, :]
                                              
                    #consider only the 16 largest correlations for each word
                    loc = np.argmax(SAC, axis = 1)               
                    
                    #find which word has largest correlation in each of these 16 cases
                    success[k] = np.mean(loc == correct_word)
                
        #find success rate (will be k/16 for some k=0,1,...,16)
           
        #average over files and correct for guessing
        cprime = (6/5) * (np.mean(success) - (1/6));
   
        #no affine transformation needed
        phi_hat = cprime;
        return phi_hat, success
             
def padSpeech(s):
#This function pads given speech vector s 
#to a minimum length allowable length of 42000
    
    #minimum speech vector length
    minLen = 42000
       
    #get length of speech vector
    l = s.size
    if l < minLen:
        #fill in zeros at the end
        size_of_S = minLen - l
        S = np.zeros(size_of_S)
        s = np.concatenate((s, S), axis= None);
        
    else:
        pass
    
    return s

def T_to_TF(x):
#This function generates a time-frequency representation for x using
#the length 512 periodic Hann window, 75% window overlap, and FFT
# - returns only the first 215 values
# - x must be column vector
# - zero padding is used if necessary to create samples for final full window.
# - window length must be evenly divisible by 4  
    m = x.size
    n = 512
    nframes = math.ceil((m - n) / (n / 4)) + 1
    newm = int((nframes - 1) * (n / 4) + n)
    
    x = np.concatenate((x, np.zeros((newm - m))))
    X = np.zeros((n, nframes))

    win = np.multiply(0.5, np.subtract(1, np.cos(np.multiply((np.conjugate(np.arange(0, 512)).T / 512), (math.pi * 2))))) #periodic Hann window
    
    for i in range(nframes):
        start = int(((i) * (n / 4)))   
        X[:, i] = np.multiply(x[start:(start + n)], win)   
    
    X = np.fft.fft(X, axis = 0) 
    X = X[0:215,:]
    
    return X

def TFnorm(X):
#This function removes the mean of every row of TF representation
#and scales each row so sum of squares is 1
    n = X.shape[1]
    
    div = np.true_divide(np.sum(X, axis = 1, keepdims = True), n) 
    div = np.reshape(div, (div.shape[0], 1))
    
    X = np.subtract(X, (div @ np.ones((1, n))))
    
    temp = np.sqrt(np.sum((X ** 2), axis = 1, keepdims = True))
    temp = np.reshape(temp, (temp.shape[0],1))
    temp = np.matlib.repmat(temp, 1, n)
    
    Y = np.true_divide(X, temp)
    
    return Y

def group_corr(X,R):
#This function uses all rows of X and R together in a cross-correlation
# - number of rows in X and R must match
# - X must have no fewer columns than R
# - evaluates all possible alignments of R with X
# - returns the shift that maximizes correlation value
# - if R has q columns then a shift value s means that R is best
#   aligned with X(:,s+1:s+q)
# - assumes R is already normalized for zero mean in each row and 
#   each row has sum of squares = 1
    
    n = X.shape[1]
    q = R.shape[1]
    
    nshifts = n - q + 1
    C = np.zeros((nshifts, 1))
    
    for i in range(nshifts):
        T = X[:, i:(i+q)]
        
        temp = np.true_divide(np.sum(T, axis = 1, keepdims = True), q)
        
        T = np.subtract(T,np.matlib.repmat(temp,1,q))
        
        kk = np.sqrt(np.sum(np.power(T,2), axis = 1, keepdims = True))
        kk_tiled = np.matlib.repmat(kk, 1, q)
        
        T = np.true_divide(T, kk_tiled)
        
        C[i] = np.sum(np.multiply(T, R), keepdims = True)
        
    shift = np.argmax(C)
    shift = shift - 1
    
    return shift

