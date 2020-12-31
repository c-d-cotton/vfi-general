#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

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
    from vfi_1endogstate_func import reshapearrays_before
    rewardarray, transmissionarray, shapestates, shapecontrols, ns, nc = reshapearrays_before(rewardarray, transmissionarray)
    outputdict['shapestates'] = shapestates
    outputdict['shapecontrols'] = shapecontrols
    outputdict['ns'] = ns
    outputdict['nc'] = nc

    # solve out vfi
    from vfi_1endogstate_func import solvevfi
    V, pol = solvevfi(rewardarray, transmissionarray, beta, printinfo = printinfo)
    outputdict['V'] = V
    outputdict['pol'] = pol

    # add transmissionstararray (don't want to carry over full transmissionarray into outputdict)
    # so makes sense to do this here
    from vfi_1endogstate_func import gentransmissionstararray
    outputdict['transmissionstararray'] = gentransmissionstararray(transmissionarray, outputdict['pol'], outputdict['ns'], outputdict['nc'])

    # add polbycontrol
    from vfi_1endogstate_func import genpolbycontrol
    outputdict['polbycontrol'] = genpolbycontrol(outputdict['pol'], outputdict['shapestates'], outputdict['shapecontrols'])
    outputdict['polbycontrolwide'] = np.reshape(outputdict['polbycontrol'], outputdict['shapestates'] + [len(outputdict['shapecontrols'])])

    return(outputdict)

