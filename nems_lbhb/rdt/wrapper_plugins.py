import numpy as np

from nems.registry import xform, xmodule


@xform()
def rdtld(key):

    xfspec = [['nems_lbhb.rdt.io.load_recording', {}]]
    return xfspec


@xform()
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
    val_dual_only = False
    val_rep_only = False

    for op in ops[1:]:
        if op.startswith('j'):
            jackknifed_fit = True
        elif op == 'ns':
            val_dual_only = True
        elif op == 'rep':
            val_rep_only = True
        else:
            njacks = int(op)

    xfspec = [['nems_lbhb.rdt.preprocessing.split_est_val',
               {'njacks': njacks, 'jackknifed_fit': jackknifed_fit,
                'val_rep_only': val_rep_only,
                'val_dual_only': val_dual_only}]]
    return xfspec


@xform()
def rdtfmt(key):

    xfspec = [['nems_lbhb.rdt.xforms.format_keywordstring', {}]]
    return xfspec


@xform()
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
