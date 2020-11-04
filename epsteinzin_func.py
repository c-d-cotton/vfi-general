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

import numpy as np

def vf_epsteinzin(rra, invies, c, BETA, V_overexog, probexog):
    """
    Value function needed to implement epstein-zin preferences.

    V_overexog is a vector of the different potential utilities receives given the value of the exogenous variable.

    Decided to remove the (1 - BETA) in front of the current period utility since otherwise I have to remember to place (1 - BETA)^{\frac{1}{1 - ies}} before the final period utility which is easy to forget and because it's more common to write value functions this way
    """
    # V = (  (1 - BETA) * c**(1-invies) + BETA * ((V_overexog ** (1 - rra)).dot(probexog)) ** ((1-invies)/(1-rra))  ) ** (1/(1-invies))
    V = (  c**(1-invies) + BETA * ((V_overexog ** (1 - rra)).dot(probexog)) ** ((1-invies)/(1-rra))  ) ** (1/(1-invies))

    return(V)


