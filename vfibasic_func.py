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

# Basic VFI Functions:{{{1
def reshapearrays_before(rewardarray, transmissionarray):
    """
    Takes arrays where I have listed states and controls across multiple dimensions and converts them into arrays where the states and controls take one dimension.
    Thus, outputs 2-dimension rewardarray, 3-dimension transmissionarray.
    Also outputs original shape of rewardarray and transmissionarray. Allows me to reshape after.
    """

    shapetransmissionarray = np.shape(transmissionarray)
    shaperewardarray = np.shape(rewardarray)
    ns = len(shapetransmissionarray) - len(shaperewardarray)
    nc = len(shaperewardarray) - ns
    shapestates = list(shaperewardarray[0: ns])
    shapecontrols = list(shaperewardarray[ns: ])

    import functools
    # multiply state dimensions together
    nsvalues = functools.reduce(lambda x, y: x*y, shapetransmissionarray[0: ns])
    # multiplying control dimensions together
    ncvalues = functools.reduce(lambda x, y: x*y, shapetransmissionarray[ns: (ns + nc)])

    rewardarray = np.reshape(rewardarray, (nsvalues, ncvalues))
    transmissionarray = np.reshape(transmissionarray, (nsvalues, ncvalues, nsvalues))

    return(rewardarray, transmissionarray, shapestates, shapecontrols, nsvalues, ncvalues)


def solvevfi(rewardarray, transmissionarray, beta, V = None, crit = 0.00001, printinfo = False, basicchecks = True):
    """
    V is the starting value.
    Other stuff is obvious from name.
    """
    import sys

    shapetransmissionarray = np.shape(transmissionarray)
    ns = shapetransmissionarray[0]
    nc = shapetransmissionarray[1]

    if basicchecks is True:
        # check the transmission probabilities make sense
        for state in range(ns):
            for action in range(nc):
                if abs(np.sum(transmissionarray[state][action][:]) - 1) > 1e6:
                    raise ValueError('ERROR: transmissionarray rows not sum to 1. State: ' + str(state) + '. Action: ' + str(action) + '. Sum: ' + str(np.sum(transmissionarray[state][action][:])) + '.')
            

    # initial values
    if V is None:
        V = np.zeros([ns])

    # note that pol needs to be integers
    pol = [0] * ns
    Vp = np.empty([ns])

    iterationi = 1
    while True:
        for s in range(0, ns):
            maxval = rewardarray[s] + beta*np.dot(transmissionarray[s], V)
            pol[s] = np.argmax(maxval)
            Vp[s] = maxval[pol[s]]

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


def genpolbycontrol(pol, shapestates, shapecontrols):
    import functools

    lenstates = len(pol)
    numcontrols = len(shapecontrols)
    
    polbycontrol = np.empty([lenstates, numcontrols], dtype = int)
    for statei in range(0, lenstates):
        # start with the chosen policy number
        remainder = pol[statei]
        for controli in range(0, numcontrols):
            # just take remainder if last control
            if controli != numcontrols - 1:
                # if controli = 0 and shapecontrols = [2,10,5], productofhigherterms = 50
                productofhigherterms = functools.reduce(lambda x, y: x*y, shapecontrols[(controli + 1): ])
                quotient = remainder // productofhigherterms
                remainder = remainder - (quotient * productofhigherterms)
            else:
                quotient = remainder

            polbycontrol[statei, controli] = quotient


    # V = np.reshape(V, tuple(shapestates))
    # pol = np.reshape(pol, tuple(shapestates))
    # polbycontrol = np.reshape(polbycontrol, tuple(shapestates + [numcontrols]))

    return(polbycontrol)


def gentransmissionstararray(transmissionarray, pol, ns, nc):
    """
    Transmission from one state to the next probabiliy (so take into account optimal policy).
    """

    transmissionstararray = np.empty([ns, ns])
    for s in range(ns):
        transmissionstararray[s] = transmissionarray[s, pol[s], :]


    return(transmissionstararray)


# Summary Functions:{{{1
def fullvfi(rewardarray, transmissionarray, beta, printinfo = False):
    """
    Does all steps of vfi basic iteration at the same time.
    """

    outputdict = {}

    # have to reshape arrays before vfi
    rewardarray, transmissionarray, shapestates, shapecontrols, ns, nc = importattr(__projectdir__ / Path('vfi_1endogstate_func.py'), 'reshapearrays_before')(rewardarray, transmissionarray)
    outputdict['shapestates'] = shapestates
    outputdict['shapecontrols'] = shapecontrols
    outputdict['ns'] = ns
    outputdict['nc'] = nc

    # solve out vfi
    V, pol = importattr(__projectdir__ / Path('vfi_1endogstate_func.py'), 'solvevfi')(rewardarray, transmissionarray, beta, printinfo = printinfo)
    outputdict['V'] = V
    outputdict['pol'] = pol

    # add transmissionstararray (don't want to carry over full transmissionarray into outputdict)
    # so makes sense to do this here
    outputdict['transmissionstararray'] = importattr(__projectdir__ / Path('vfi_1endogstate_func.py'), 'gentransmissionstararray')(transmissionarray, outputdict['pol'], outputdict['ns'], outputdict['nc'])

    # add polbycontrol
    outputdict['polbycontrol'] = importattr(__projectdir__ / Path('vfi_1endogstate_func.py'), 'genpolbycontrol')(outputdict['pol'], outputdict['shapestates'], outputdict['shapecontrols'])
    outputdict['polbycontrolwide'] = np.reshape(outputdict['polbycontrol'], outputdict['shapestates'] + [len(outputdict['shapecontrols'])])

    return(outputdict)

