#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

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


