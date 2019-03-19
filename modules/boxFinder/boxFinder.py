import re

# Function to get E.coli promotor sigma70 box information
# Retrun value consist of a list with:
# 1. Boolean if -35 box was found
# 2. Boolean if extended -10 box was found
# 3. Boolean if -10 box was found
# 4. distance between -35 and (ext)-10 if one of the boxes was not found a 0 is returned
def boxFinder(seq):
    box10 = findRe(pattern='TA[ACGT]{3}AT',seq=seq.upper()[35:60],start=True, add = 35)
    box35 = findRe(pattern='TT{2}[GT][ACGT]{3}',seq=seq.upper()[:35],start=False)
    box10Ext = False
    afstand = 0
    if not isinstance(box10,bool):
        box10Ext = findRe(pattern='G[ACTG]TA[ACGT]{3}AT',seq=seq.upper()[35:60],start=True, add = 35)
        if not isinstance(box35,bool):
            if not isinstance(box10Ext,bool):
                afstand = box10Ext - box35
            else:
                afstand = box10 - box35

    return ([not isinstance(box35,bool),
             not isinstance(box10Ext,bool),
             not isinstance(box10,bool),
             afstand])


def findRe(pattern, seq, start, add = 0):
    try:
        x = re.search(pattern,seq)
        if start:
            return(x.start()+add)
        else:
            return(x.end()+add)
    except AttributeError as a:
        return(False)
