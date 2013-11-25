#!/usr/bin/env python

import scipy as sp

if __name__ == '__main__':
    l = sp.loadtxt('labels.dat', dtype=int)
    lp = sp.zeros((l.size, 10), dtype=int)
    lp[:, l-1] = sp.ones((l.size, 1), dtype=int)
    sp.savetxt('labels_pos.dat', lp)

