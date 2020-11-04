#!/usr/bin/env python3
# PYTHON_PREAMBLE_START_STANDARD:{{{

# Christopher David Cotton (c)
# http://www.cdcotton.com

# modules needed for preamble
import importlib
import os
from pathlib import Path
import sys

# Get full real filename
__fullrealfile__ = os.path.abspath(__file__)

# Function to get git directory containing this file
def getprojectdir(filename):
    curlevel = filename
    while curlevel is not '/':
        curlevel = os.path.dirname(curlevel)
        if os.path.exists(curlevel + '/.git/'):
            return(curlevel + '/')
    return(None)

# Directory of project
__projectdir__ = Path(getprojectdir(__fullrealfile__))

# Function to call functions from files by their absolute path.
# Imports modules if they've not already been imported
# First argument is filename, second is function name, third is dictionary containing loaded modules.
modulesdict = {}
def importattr(modulefilename, func, modulesdict = modulesdict):
    # get modulefilename as string to prevent problems in <= python3.5 with pathlib -> os
    modulefilename = str(modulefilename)
    # if function in this file
    if modulefilename == __fullrealfile__:
        return(eval(func))
    else:
        # add file to moduledict if not there already
        if modulefilename not in modulesdict:
            # check filename exists
            if not os.path.isfile(modulefilename):
                raise Exception('Module not exists: ' + modulefilename + '. Function: ' + func + '. Filename called from: ' + __fullrealfile__ + '.')
            # add directory to path
            sys.path.append(os.path.dirname(modulefilename))
            # actually add module to moduledict
            modulesdict[modulefilename] = importlib.import_module(''.join(os.path.basename(modulefilename).split('.')[: -1]))

        # get the actual function from the file and return it
        return(getattr(modulesdict[modulefilename], func))

# PYTHON_PREAMBLE_END:}}}

import copy
import functools
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fminbound 

# Alternative VFI - One State is Control:{{{1
def solvevfi_1endogstate_discrete(rewardarray, transmissionarray, beta, V = None, crit = 0.000001, printinfo = False, basicchecks = True, Vfunc = None, functiontype = None):
    """
    Uses a lot less states and is thus a lot quicker than solvevfi function.

    Agents pick state1' today and then state1' becomes state1 at time t + 1.
    state2 is another set of states.
    transmissionarray is only over state2 since state1' is picked at time t.
    "Discrete" in the sense that I only let agents pick discrete values state1'.

    rewardarray.shape = ns1 x ns2 x ns1
    transmissionarray = ns2 x ns2 where (i,j) is P(ns2' = j|ns2 = i)
    (V = ns1 x ns2). So (i,j) of V is V(s1 = ith element, s2 = jth element)

    Vfunc allows me to get non-standard forms of value function i.e. Epstein-Zin
    If specify Vfunc, need to specify functiontype = 'value-full'/'value-betaEV'
    By default, use 'value-full'
    In this case, rewardarray really just represents any function of s1, s2, s1prime i.e. it could be consumption rather than utility(consumption)
    """

    shaperewardarray = np.shape(rewardarray)
    ns1 = shaperewardarray[0]
    ns2 = shaperewardarray[1]
    
    if basicchecks is True:
        # check the transmission probabilities make sense
        for state in range(ns2):
            if abs(np.sum(transmissionarray[state][:]) - 1) > 1e6:
                raise ValueError('ERROR: transmissionarray rows not sum to 1. State: ' + str(state) + '. Sum: ' + str(np.sum(transmissionarray[state][action][:])) + '.')
            

    # initial values
    if V is None:
        # set to be ones to begin in case V is to some power 
        V = np.ones([ns1, ns2])
    # pol must be integers
    pol = np.zeros([ns1, ns2])
    pol = pol.astype(int)
    Vp = copy.deepcopy(V)

    if Vfunc is not None and functiontype is None:
        functiontype = 'value-full'

    iterationi = 1
    while True:
        for s1old in range(0, ns1):
            for s2 in range(0, ns2):
                if Vfunc is None:
                    # standard case with standard V form
                    maxval = rewardarray[s1old, s2, :] + beta*V.dot(transmissionarray[s2, :])
                else:
                    # specified a function to use for the value function
                    # don't do this with vector form so specify empty maxval first
                    maxval = np.empty(ns1)
                    for s1prime in range(0, ns1):
                        if functiontype == 'value-full':
                            maxval[s1prime] = Vfunc(beta, V[s1prime, :], transmissionarray[s2, :], rewardarray[s1old, s2, s1prime])
                        elif functiontype == 'value-betaEV':
                            maxval[s1prime] = Vfunc(beta*V.dot(transmissionarray[s2, :]), rewardarray[s1old, s2, s1prime])
                        else:
                            raise ValueError('functiontype not specified correctly.')
                        

                pol[s1old, s2] = np.argmax(maxval)
                Vp[s1old, s2] = maxval[pol[s1old, s2]]

        diff = np.max(np.abs(Vp - V))
        if np.isnan(diff):
            print('ERROR: diff is nan on iteration ' + str(iterationi))
            sys.exit(1)
        if printinfo is True:
            print('Iteration ' + str(iterationi) + '. Diff: ' + str(diff) + '.')
        iterationi = iterationi + 1
        if diff < crit:
            break
        else:
            # need copy otherwise when replace Vp[s], V[s] also updates
            V = Vp.copy()


    return(Vp, pol)


def solvevfi_1endogstate_continuous(inputfunction, endogstatevec, exogstatevec, transmissionarray, beta, V = None, crit = 0.000001, printinfo = False, basicchecks = True, s1low = None, s1high = None, boundfunction = None, functiontype = 'reward', interpmethod = 'numpy'):
    """
    Agents pick continuous state1' today and then state1' becomes state1 at time t + 1.
    Calculate value of future by interpolation.
    state2 is another set of states.
    transmissionarray is only over state2 since state1' is picked at time t.
    "Discrete" in the sense that I only let agents pick discrete values state1'.

    endogstatevec and exogstatevec are the actual values of the states - I need these to input in the reward function
    transmissionarray = ns2 x ns2 where (i,j) is P(ns2' = j|ns2 = i)
    (V = ns1 x ns2). So (i,j) of V is V(s1 = ith element, s2 = jth element)
    if inputfunction == 'reward': inputfunction is the rewardfunction with the following arguments: s1_old-val, s2_val, s1_new_val
    if inputfunction == 'value-betaEV': inputfunction is the valuefunction with the following arguments: betaEVfunc, s1_old_val, s2_val, s1_new_val. betaEVfunc is a function over s1_new_val which is defined during the iteration.
    if inputfunction == 'value-full': inputfunction is the valuefunction with the following arguments: betaval, Vfunc, nextperiodprobs, s1_old_val, s2_val, s1_new_val. Vfunc is a function over s1_new_val which is defined during the iteration.

    boundfunction: Input endogenous and exogenous state and return tuple of length 2 with lower and upper bound. Give None in no relevant lower or upper bound. If no boundfunction is specified, I just use the endogenous limits as my limiting choices.
    """

    ns1 = len(endogstatevec)
    ns2 = np.shape(transmissionarray)[0]
    
    if basicchecks is True:
        # check the transmission probabilities make sense
        for state in range(ns2):
            if abs(np.sum(transmissionarray[state][:]) - 1) > 1e6:
                raise ValueError('ERROR: transmissionarray rows not sum to 1. State: ' + str(state) + '. Sum: ' + str(np.sum(transmissionarray[state][action][:])) + '.')
            

    # initial values
    if V is None:
        # don't start with zero in case V is to the power of something
        V = np.ones([ns1, ns2])

    pol = np.empty([ns1, ns2])
    Vp = copy.deepcopy(V)

    # define negative value function
    if functiontype == 'reward':
        def negativevaluefunction(betaEVfunc, s1_old_val, s2_val, s1_new_val):
            value = inputfunction(s1_old_val, s2_val, s1_new_val) + betaEVfunc(s1_new_val)
            return(-value)
    elif functiontype == 'value-betaEV':
        def negativevaluefunction(betaEVfunc, s1_old_val, s2_val, s1_new_val):
            return(-inputfunction(betaEVfunc, s1_old_val, s2_val, s1_new_val))
    elif functiontype == 'value-full':
        def negativevaluefunction(betaval, Vfunc, nextperiodprobs, s1_old_val, s2_val, s1_new_val):
            return(-inputfunction(betaval, Vfunc, nextperiodprobs, s1_old_val, s2_val, s1_new_val))
    else:
        raise ValueError('function type procedure not defined.')
        
        
    iterationi = 1
    while True:
        for s1_old in range(0, ns1):
            for s2 in range(0, ns2):

                # if using value-full then I input V directly into the negativevaluefunction argument
                # otherwise input betaEV
                if functiontype == 'value-full':

                    if interpmethod == 'numpy':
                        def Vfunc(s1_new_val):
                            # in one dimension, interpedV = np.interp(s1_new_val, endogstatevec, Vp)
                            interpedV = np.empty(ns2)
                            for s2_2 in range(ns2):
                                interpedV[s2_2] = np.interp(s1_new_val, endogstatevec, Vp[:, s2_2])
                            return(interpedV)
                    elif interpmethod == 'scipy':
                        Vfunc = interp1d(endogstatevec, Vp, axis = 0)
                    else:
                        raise ValueError('interpmethod misspecified')

                    nextperiodprobs = transmissionarray[s2, :]
                else:
                    # compute expected value function
                    betaEV = beta*Vp.dot(transmissionarray[s2, :])

                    if interpmethod == 'numpy':
                        def betaEVfunc(s1_new_val):
                            return(np.interp(s1_new_val, endogstatevec, betaEV))
                    elif interpmethod == 'scipy':
                        betaEVfunc = interp1d(endogstatevec, betaEV, axis = 0)
                    else:
                        raise ValueError('Misspecified interpmethod')

                # get general func to minimize from
                if functiontype == 'value-full':
                    currentnegvalfunc = functools.partial(negativevaluefunction, beta, Vfunc, nextperiodprobs, endogstatevec[s1_old], exogstatevec[s2])
                else:
                    currentnegvalfunc = functools.partial(negativevaluefunction, betaEVfunc, endogstatevec[s1_old], exogstatevec[s2])

                # get bounds
                s1low = None
                s1high = None
                if boundfunction is not None:
                    s1low, s1high = boundfunction(endogstatevec[s1_old], exogstatevec[s2])
                if s1low is None:
                    s1low = endogstatevec[0]
                if s1low < endogstatevec[0]:
                    s1low = endogstatevec[0]
                if s1high is None:
                    s1high = endogstatevec[-1]
                if s1high > endogstatevec[-1]:
                    s1high = endogstatevec[-1]

                # get s1new that maximises utility
                pol[s1_old, s2] = fminbound(currentnegvalfunc, float(s1low), float(s1high))

                # needs to be negative since was using negative utility
                Vp[s1_old, s2] = -currentnegvalfunc(pol[s1_old, s2])

        diff = np.max(np.abs(Vp - V))
        if np.isnan(diff):
            print('ERROR: diff is nan on iteration ' + str(iterationi))
            sys.exit(1)
        if printinfo is True:
            print('Iteration ' + str(iterationi) + '. Diff: ' + str(diff) + '.')
        iterationi = iterationi + 1
        if diff < crit:
            break
        else:
            # need copy otherwise when replace Vp[s], V[s] also updates
            V = Vp.copy()
    
    return(Vp, pol)


# Get PolProbs:{{{1
def getpolprobs_1endogstate_discrete(pol, ns1prime = None):
    """
    For when I have that pol returns a continuous value i.e. pol[s1, s2] returns a continuous value rather than the index of s1' or the value of s1'.
    function rewrites pol as polprobs i.e. prob that if you start with s1, s2 that you'll get back s1', s2'

    This is different to the discrete transmissionstararray because it returns a vector of probabilities for each (s1,s2) pair rather than a single index
    """
    ns1 = np.shape(pol)[0]
    ns2 = np.shape(pol)[1]

    if ns1prime is None:
        ns1prime = ns1

    polprobs = np.zeros([ns1, ns2, ns1prime])
    for s1 in range(ns1):
        for s2 in range(ns2):
            # assign 1 probability to s1, s2
            polprobs[s1, s2, pol[s1, s2]] = 1

    return(polprobs)


def getpolprobs_1endogstate_continuous(pol, endogstatevec):
    """
    For when I have that pol returns a continuous value i.e. pol[s1, s2] returns a continuous value rather than the index of s1' or the value of s1'.
    function rewrites pol as polprobs i.e. prob that if you start with s1, s2 that you'll get back s1', s2'

    This is different to the discrete transmissionstararray because it returns a vector of probabilities for each (s1,s2) pair rather than a single index
    """
    ns1 = np.shape(pol)[0]
    ns2 = np.shape(pol)[1]
    ns1prime = len(endogstatevec)

    polprobs = np.empty([ns1, ns2, ns1prime])
    for s1 in range(ns1):
        for s2 in range(ns2):
            # returns probability distribution of potential s1' choices
            polprobs[s1, s2, :] = importattr(__projectdir__ / Path('submodules/python-math-func/dist_func.py'), 'weightvaluediscretevec')(pol[s1, s2], endogstatevec)

    return(polprobs)


# State Dists - With Transmissionstararray:{{{1
def gentransmissionstararray_1endogstate_discrete(transmissionarray, pol, ns1prime = None):
    """
    Transmission from one state to the next probabiliy (so take into account optimal policy).
    """

    ns1 = np.shape(pol)[0]
    ns2 = np.shape(pol)[1]
    if ns1prime is None:
        ns1prime = ns1
    ns2prime = np.shape(transmissionarray)[1]

    # reverse order of ns2, ns1 since this makes it easier to work with
    transmissionstararray = np.zeros([ns1 * ns2, ns1prime * ns2prime])

    # replace only the part of each row of transmissionstararray with the transmissionstararray given s2 in the position of s1 that is chosen given s1,s2
    for s1 in range(0, ns1):
        for s2 in range(0, ns2):
            transmissionstararray[s1 * ns2 + s2, pol[s1,s2] * ns2prime: (pol[s1,s2] + 1) * ns2prime] = transmissionarray[s2, :]
        
    return(transmissionstararray)


def gentransmissionstararray_1endogstate_polprobs(transmissionarray, polprobs):
    """
    Transmission from one state to the next probability (so take into account optimal policy).
    Note that transmissionstararray is of dimension s1xs2,s1xs2 where the probability of moving from states (0,1) to states (0,2) is given by (1,2).
    """

    ns1 = np.shape(polprobs)[0]
    ns2 = np.shape(polprobs)[1]
    ns1prime = np.shape(polprobs)[2]
    ns2prime = np.shape(transmissionarray)[1]

    transmissionstararray = np.zeros([ns1 * ns2, ns1prime * ns2prime])

    # replace only the part of each row of transmissionstararray with the transmissionstararray given s2 in the position of s1 that is chosen given s1,s2
    for s1 in range(0, ns1):
        for s2 in range(0, ns2):
            for s1new in range(0, ns1prime):
                transmissionstararray[s1 * ns2 + s2, s1new * ns2prime: (s1new + 1) * ns2prime] = polprobs[s1, s2, s1new] * transmissionarray[s2, :]
        
    return(transmissionstararray)


def getstationarydist_1endogstate_full(transmissionstararray, ns1, crit = 1e-9):
    """
    Just find the stationary distribution for the chosen state, not the exogenous state.

    By default only find the aggregated distribution for the first state.
    """

    dist = importattr(__projectdir__ / Path('submodules/python-math-func/markov_func.py'), 'getstationarydist')(transmissionstararray, crit = crit)

    ns = np.shape(transmissionstararray)[0]
    ns2 = int(ns/ns1)

    # put into matrix
    fullstatedist = dist.reshape([ns1, ns2])
    
    # average by row
    endogstatedist = fullstatedist.sum(axis = 1)

    return(fullstatedist, endogstatedist)
            

# State Dists - Without Transmissionstararray:{{{1
def getstationarydist_1endogstate_direct(transmissionarray, polprobs, distinit = None, crit = 1e-9):
    """
    getstationarydist_1endogstate_full can take a long time as I have to generate a big matrix and then multiply a lot of times
    This is not necessary because the probability that I end up in a given state is exogenous

    dist is a matrix with dist_{i,j} representing the probability of having endog state i and exog state j
    """

    ns1 = np.shape(polprobs)[0]
    ns2 = np.shape(transmissionarray)[0]

    # starting distribution
    if distinit is not None:
        dist = distinit
    else:
        dist = np.ones((ns1, ns2)) / (ns1 * ns2)

    newdist = np.empty([ns1, ns2])

    while True:
        # solve for s1', s2 given s1, s2
        for s2 in range(ns2):
            newdist[:, s2] = dist[:, s2].dot(polprobs[:, s2, :])

        # solve for s1', s2' given s1', s2
        newdist = newdist.dot(transmissionarray)

        diff = np.max(np.abs(newdist - dist))

        dist = copy.deepcopy(newdist)
        if diff < crit:
            break

    # average by row
    endogstatedist = dist.sum(axis = 1)

    return(dist, endogstatedist)


