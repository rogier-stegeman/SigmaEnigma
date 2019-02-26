import re

def boxFinder(seq):
    box10 = findRe(pattern='[TG]A[ACGT]{3}[AT]',seq=seq.upper()[35:60],start=True, add = 35)
    box35 = findRe(pattern='T[GT]{2}[ACGT]{3}',seq=seq.upper()[:35],start=False)
    box10Ext = False
    afstand = 0
    if not isinstance(box10,bool):
        box10Ext = findRe(pattern='G[ACTG][TG]A[ACGT]{3}[AT]',seq=seq.upper()[35:60],start=True, add = 35)
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
