import numpy as np


def rdtld(key):

    xfspec = [['nems_lbhb.rdt.io.load_recording', {}]]
    return xfspec


def rdtsev(key):
    """
    :param key: rdtsev.<op1>.<op2>
     op == 'j' : preserve all est/val subsets for jackknife
       (default is to discard all but the first)
     op == int : set njacks = int(op)
    :return: xfspec call to rdt.preprocessing.split_est_val
    """
    ops = key.split(".")
    njacks=5
    jackknifed_fit = False
    for op in ops[1:]:
        if op.startswith('j'):
            jackknifed_fit = True
        else:
            njacks = int(op)

    xfspec = [['nems_lbhb.rdt.preprocessing.split_est_val',
               {'njacks': njacks, 'jackknifed_fit': jackknifed_fit}]]
    return xfspec


def rdtfmt(key):

    xfspec = [['nems_lbhb.rdt.xforms.format_keywordstring', {}]]
    return xfspec

def rdtshf(key):
    ops = key.split(".")
    shuff_streams=False
    shuff_rep=False
    for op in ops:
        if op=="str":
            shuff_streams=True
        if op=="rep":
            shuff_rep=True

    xfspec = [['nems_lbhb.rdt.preprocessing.rdt_shuffle',
               {'shuff_rep': shuff_rep, 'shuff_streams': shuff_streams}]]
    return xfspec
