#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import functools
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fminbound 

# General Functions:{{{1
def Vprime_get(lastperiodutility, endogstate_last, exogstate_last):
    Vprime = np.empty([len(endogstate_last), len(exogstate_last)])
    for i in range(len(endogstate_last)):
        for j in range(len(exogstate_last)):
            Vprime[i, j] = lastperiodutility(endogstate_last[i], exogstate_last[j])

    return(Vprime)
        

# Value Function Solve Back - One State is Control:{{{1
def vf_1endogstate_discrete_oneiteration(rewardarray, Vprime, transmissionarray, beta, basicchecks = True, Vfunc = None, functiontype = None, V = None, t = None):

    shaperewardarray = np.shape(rewardarray)
    ns1 = shaperewardarray[0]
    ns2 = shaperewardarray[1]
    ns1prime = np.shape(Vprime)[0]
    ns2prime = np.shape(Vprime)[1]

    
    if basicchecks is True:
        # check ns2 correct
        if np.shape(transmissionarray)[0] != ns2:
            raise ValueError('Number of rows incorrect in transmission array.')
        # check ns2prime correct
        if np.shape(transmissionarray)[1] != ns2prime:
            raise ValueError('Number of columns incorrect in transmission array.')

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

    if Vfunc is not None and functiontype is None:
        functiontype = 'value-full'

    for s1old in range(0, ns1):
        for s2 in range(0, ns2):
            if Vfunc is None:
                # standard case with standard V form
                maxval = rewardarray[s1old, s2, :] + beta*Vprime.dot(transmissionarray[s2, :])
            else:
                # specified a function to use for the value function
                # don't do this with vector form so specify empty maxval first
                maxval = np.empty(ns1prime)
                for s1prime in range(0, ns1prime):
                    if functiontype == 'value-full':
                        maxval[s1prime] = Vfunc(beta, Vprime[s1prime, :], transmissionarray[s2, :], rewardarray[s1old, s2, s1prime])
                    elif functiontype == 'value-betaEV':
                        maxval[s1prime] = Vfunc(beta*Vprime.dot(transmissionarray[s2, :]), rewardarray[s1old, s2, s1prime])
                    else:
                        raise ValueError('functiontype not specified correctly.')
                    

            pol[s1old, s2] = np.argmax(maxval)
            V[s1old, s2] = maxval[pol[s1old, s2]]

    return(V, pol)


def vf_1endogstate_continuous_oneiteration(inputfunction, Vprime, endogstate_now, endogstate_future, exogstate_now, exogstate_future, transmissionarray, beta, basicchecks = True, boundfunction = None, t = None, functiontype = 'reward', interpmethod = 'numpy'):
    """
    Given value function in next period find policy function and value function today.

    t argument just allows me to check for errors more easily - isn't directly included in function.

    if inputfunction == 'reward': inputfunction is the rewardfunction with the following arguments: s1_old-val, s2_val, s1_new_val
    if inputfunction == 'value-betaEV': inputfunction is the valuefunction with the following arguments: betaEVfunc, s1_old_val, s2_val, s1_new_val. betaEVfunc is a function over s1_new_val which is defined during the iteration.
    if inputfunction == 'value-full': inputfunction is the valuefunction with the following arguments: betaval, Vfunc, nextperiodprobs, s1_old_val, s2_val, s1_new_val. Vfunc is a function over s1_new_val which is defined during the iteration.

    if usebetaEV is True then just compute beta * EV in the usual way here rather than having to incorporate it in the valuefunction I have to input
    if usebetaEV is False then I have to include the next period value function part in my valuefunction using BETA, Vfunc and nextperiodprobs (Vfunc is over the domain of s1_val_new and returns a vector of future values depending on s2_val_new and nextperiodprobs returns probabilities of s2_val_new (which is independent of choice of s1_val_new by assumption)).

    exogstate_future actually unusued - just used to do a check atm
    """
    ns1 = len(endogstate_now)
    ns2 = len(exogstate_now)

    if ns2 != np.shape(transmissionarray)[0]:
        raise ValueError('Number of transmission array rows not the same as length of exogstate_now')
    if len(exogstate_future) != np.shape(transmissionarray)[1]:
        raise ValueError('Number of transmission array columns not the same as length of exogstate_future')
    
    if basicchecks is True:
        # check the transmission probabilities make sense
        for state in range(ns2):
            if abs(np.sum(transmissionarray[state][:]) - 1) > 1e6:
                raise ValueError('ERROR: transmissionarray rows not sum to 1. State: ' + str(state) + '. Sum: ' + str(np.sum(transmissionarray[state][action][:])) + '.')
            
    V = np.empty([ns1, ns2])
    pol = np.empty([ns1, ns2])

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
                            interpedV[s2_2] = np.interp(s1_new_val, endogstate_future, Vprime[:, s2_2])
                        return(interpedV)
                elif interpmethod == 'scipy':
                    Vfunc = interp1d(endogstate_future, Vprime, axis = 0)
                else:
                    raise ValueError('interpmethod misspecified')

                nextperiodprobs = transmissionarray[s2, :]
            else:
                # compute expected value function
                betaEV = beta*Vprime.dot(transmissionarray[s2, :])

                if interpmethod == 'numpy':
                    def betaEVfunc(s1_new_val):
                        return(np.interp(s1_new_val, endogstate_future, betaEV))
                elif interpmethod == 'scipy':
                    betaEVfunc = interp1d(endogstate_future, betaEV)
                else:
                    raise ValueError('interpmethod misspecified')


            # get general func to minimize from
            if functiontype == 'value-full':
                currentnegvalfunc = functools.partial(negativevaluefunction, beta, Vfunc, nextperiodprobs, endogstate_now[s1_old], exogstate_now[s2])
            else:
                currentnegvalfunc = functools.partial(negativevaluefunction, betaEVfunc, endogstate_now[s1_old], exogstate_now[s2])

            # get bounds
            s1low = None
            s1high = None
            if boundfunction is not None:
                s1low, s1high = boundfunction(endogstate_now[s1_old], exogstate_now[s2])
            s1low2 = endogstate_future[0]
            s1high2 = endogstate_future[-1]
            if s1low is None:
                s1low = s1low2
            else:
                s1low = np.max([s1low, s1low2])
            if s1high is None:
                s1high = s1high2
            else:
                s1high = np.min([s1high, s1high2])
                
            # get s1new that maximises utility
            pol[s1_old, s2] = fminbound(currentnegvalfunc, float(s1low), float(s1high))

            # needs to be negative since was using negative utility
            V[s1_old, s2] = -currentnegvalfunc(pol[s1_old, s2])

    return(V, pol)


def vf_solveback_discrete(rewardarray_list, Vprime, transmissionarray_list, beta_list, basicchecks = True, Vfunc_list = None, functiontype_list = None):
    """
    """

    # number of periods
    T = len(rewardarray_list) + 1

    # convert None types
    if Vfunc_list is None:
        Vfunc_list = [None] * (T - 1)
    if functiontype_list is None:
        functiontype_list = [None] * (T - 1)

    # verify correct length
    if len(transmissionarray_list) != T - 1:
        raise ValueError('transmissionarray_list should be of length T - 1.')
    if len(beta_list) != T - 1:
        raise ValueError('beta_list should be of length T - 1')
    if len(Vfunc_list) != T - 1:
        raise ValueError('Vfunc_list should be of length T - 1.')
    if len(functiontype_list) != T - 1:
        raise ValueError('functiontype_list should be of length T - 1.')


    polreversed = []
    Vreversed = [Vprime]

    for t in reversed(range(0, T - 1)):
        V, pol = vf_1endogstate_discrete_oneiteration(rewardarray_list[t], Vreversed[-1], transmissionarray_list[t], beta_list[t], basicchecks = True, t = t, Vfunc = Vfunc_list[t], functiontype = functiontype_list[t])

        polreversed.append(pol)
        Vreversed.append(V)

    pollist = list(reversed(polreversed))
    Vlist = list(reversed(Vreversed))

    return(Vlist, pollist)


def vf_solveback_continuous(inputfunction_list, lastperiodutility, endogstate_list, exogstate_list, transmissionarray_list, beta_list, basicchecks = True, boundfunction_list = None, functiontype_list = None):
    """
    For this description, periods = T including last period

    inputfunction_list should be T-1 (since last period given by lastutilityfunction)
    endogstate_list should be length T
    exogstate_list should be length T
    transmissionarray_list should be length T - 1 (since giving transmission of exogstates from 1 to 2, ..., T-1 to T
    beta_list should be length T - 1 since gives discount of 2 at 1, ..., T at T - 1

    boundfunction_list is a list of boundfunctions that allows me to specify which values I consider in the max problem
    basicchecks just verifies that the transmissionarray has correct probability features
    """

    # verify correct length
    T = len(endogstate_list)
    if len(exogstate_list) != T:
        raise ValueError('endogstate_list should be same length as exogstate_list.')
    if len(inputfunction_list) != T - 1:
        raise ValueError('endogstate_list should be one longer than inputfunction_list.')
    if len(transmissionarray_list) != T - 1:
        raise ValueError('endogstate_list should be one longer than transmissionarray_list.')
    if len(beta_list) != T - 1:
        raise ValueError('endogstate_list should be one longer than beta_list.')

    Vprime = Vprime_get(lastperiodutility, endogstate_list[-1], exogstate_list[-1])

    polreversed = []
    Vreversed = [Vprime]

    if boundfunction_list is None:
        boundfunction_list = [None] * (T - 1)
    if functiontype_list is None:
        functiontype_list = ['reward'] * (T - 1)
    if functiontype_list == 'reward':
        functiontype_list = ['reward'] * (T - 1)
    if functiontype_list == 'value-betaEV':
        functiontype_list = ['value-betaEV'] * (T - 1)
    if functiontype_list == 'value-full':
        functiontype_list = ['value-full'] * (T - 1)

    for t in reversed(range(0, T - 1)):
        V, pol = vf_1endogstate_continuous_oneiteration(inputfunction_list[t], Vreversed[-1], endogstate_list[t], endogstate_list[t + 1], exogstate_list[t], exogstate_list[t + 1], transmissionarray_list[t], beta_list[t], basicchecks = True, t = t, boundfunction = boundfunction_list[t], functiontype = functiontype_list[t])

        polreversed.append(pol)
        Vreversed.append(V)

    pollist = list(reversed(polreversed))
    Vlist = list(reversed(Vreversed))

    return(Vlist, pollist)


# Solve for Distribution Continuous Space:{{{1
def dist_solveback(startdist_endog, startdist_exog, endogstate_list, transmissionarray_list, pol_list, transmissionstarmethod = True):
    """
    Return dist_list which is just a list of the distribution of the exogenous variable at each period

    Two methods possible:
    - Compute via transmissionstar
    - Compute by updating dist for endogenous and then exogenous separately

    Works with continuous or discrete time
    Determine whether continuous policy function or not based upon whether the (0,0) element is an integer or not
    """
    T = len(pol_list) + 1

    # determine whether policy function is from a discrete/continuous space
    # do so by checking if policy function contains integers (in which case its discrete)
    if isinstance(pol_list[0][0, 0], int) or isinstance(pol_list[0][0, 0], np.int64):
        discrete = True
    else:
        discrete = False
        

    # get initdist
    nextperiodfulldist = np.kron(startdist_endog, startdist_exog)

    # we should end up with dist_list that is of length T
    fulldistlist = []
    fulldistlist.append(nextperiodfulldist.reshape((len(startdist_endog), -1)))

    endogdistlist = []
    endogdistlist.append(fulldistlist[0].sum(axis = 1))

    # len of transmissionarray_list is T - 1
    for t in range(T - 1):
        # numendogtp1
        numendogtp1 = len(endogstate_list[t + 1])

        # get the transmission star array
        if transmissionstarmethod is True:
            if discrete is True:
                from vfi_1endogstate_func import gentransmissionstararray_1endogstate_discrete
                transmissionstararray = gentransmissionstararray_1endogstate_discrete(transmissionarray_list[t], pol_list[t], ns1prime = len(endogstate_list[t + 1]))
            else:
                # need endogstate_{t + 1} since we need to find how the choice of state in the next period can be represented as endogstate_{t + 1}
                from vfi_1endogstate_func import getpolprobs_1endogstate_continuous
                polprobs_t = getpolprobs_1endogstate_continuous(pol_list[t], endogstate_list[t + 1])

                from vfi_1endogstate_func import gentransmissionstararray_1endogstate_polprobs
                transmissionstararray = gentransmissionstararray_1endogstate_polprobs(transmissionarray_list[t], polprobs_t)

            # nextperiodfulldist = transmissionstararray.dot(nextperiodfulldist)
            nextperiodfulldist = np.dot(nextperiodfulldist, transmissionstararray)

        else:
            # uses the separatestates method
            
            if discrete is True:
                from vfi_1endogstate_func import getpolprobs_1endogstate_discrete
                polprobs_t = getpolprobs_1endogstate_discrete(pol_list[t], ns1prime = len(endogstate_list[t + 1]))
            else:
                from vfi_1endogstate_func import getpolprobs_1endogstate_continuous
                polprobs_t = getpolprobs_1endogstate_continuous(pol_list[t], endogstate_list[t + 1])

            # now same steps as gentransmissionstararray_1endogstate_polprobs in vfi_1endogstate_func.py

            # getting s1' state so need t + 1 number of endogenous states
            ns1 = np.shape(endogstate_list[t + 1])[0]
            ns2 = np.shape(transmissionarray_list[t])[0]

            # solve for s1',s2 given s1,s2
            nextperiodfulldist = np.empty([ns1, ns2])
            for s2 in range(ns2):
                nextperiodfulldist[:, s2] = fulldistlist[t][:, s2].dot(polprobs_t[:, s2, :])

            # solve for s1', s2' given s1', s2
            nextperiodfulldist = nextperiodfulldist.dot(transmissionarray_list[t])



        # adjust nextperiodfulldist to get just the dist of the endogenous state and add to distmatrix
        fulldistlist.append(nextperiodfulldist.reshape((numendogtp1, -1)))
        endogdistlist.append(fulldistlist[-1].sum(axis = 1))

    return(fulldistlist, endogdistlist)


def dist_meanvar(distlist, varbystatelist):
    """
    Computes the mean of var by each period from distlist and the value that the var would take in each state in each period.

    Two options:
    - distlist can be the endogstate list i.e. just the probabilities by endogenous state in which case varbystatelist should be the values that the variable takes by each endogenous state only
    - distlist can be the fulldistlist in which case varbystatelist should be the values that the variable takes by each exogenous AND endogenous state

    Either way, distlist and varbystatelist should have the same dimensions for each period
    """
    meanvarlist = []
    for i in range(len(distlist)):
        meanvarlist.append( np.sum(np.multiply(distlist[i], varbystatelist[i])) )

    return(meanvarlist)
    
        
    
