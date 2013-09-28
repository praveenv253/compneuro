#!/usr/bin/env python

from matplotlib.pyplot import *
from scipy import *

# THIS PROGRAM DEMONSTRATES HODGKIN HUXLEY MODEL IN CURRENT CLAMP EXPERIMENTS
# AND SHOWS ACTION POTENTIAL PROPAGATION

# Time is in s, voltage in mV, conductances in mS/mm^2, capacitance in uF/mm^2

# Threshold value of current is 0.0223

#ImpCur = input('enter the value of the impulse current in microamperes: ')

import sys

if len(sys.argv) != 2:
    print 'Incorrect usage. Supply magnitude of current pulse'
    sys.exit(0)

ImpCur = float(sys.argv[1])

gkmax = .36
vk = -77
gnamax = 1.20
vna = 50
gl = 0.003
vl = -54.387
cm = .01

dt = 0.01
niter = 10000
t = arange(1, niter+1) * dt
iapp = ImpCur * ones(niter)

v = -64.9964
m = 0.0530
h = 0.5960
n = 0.3177

gnahist = zeros(niter)
gkhist = zeros(niter)
vhist = zeros(niter)
mhist = zeros(niter)
hhist = zeros(niter)
nhist = zeros(niter)

for iterator in range(niter):
    gna = gnamax * m ** 3 * h
    gk = gkmax * n ** 4
    gtot = gna + gk + gl
    vinf = ((gna * vna + gk * vk + gl * vl) + iapp[iterator]) / gtot
    tauv = cm / gtot
    v = vinf + (v - vinf) * exp(-dt / tauv)
    alpham = 0.1 * (v + 40) / (1 - exp(-(v + 40) / 10))
    betam = 4 * exp(-0.0556 * (v + 65))
    alphan = 0.01 * (v + 55) / (1 - exp(-(v + 55) / 10))
    betan = 0.125 * exp(-(v + 65) / 80)
    alphah = 0.07 * exp(-0.05 * (v + 65))
    betah = 1 / (1 + exp(-0.1 * (v + 35)))
    taum = 1 / (alpham + betam)
    tauh = 1 / (alphah + betah)
    taun = 1 / (alphan + betan)
    minf = alpham * taum
    hinf = alphah * tauh
    ninf = alphan * taun
    m = minf + (m - minf) * exp(-dt / taum)
    h = hinf + (h - hinf) * exp(-dt / tauh)
    n = ninf + (n - ninf) * exp(-dt / taun)
    vhist[iterator] = v
    mhist[iterator] = m
    hhist[iterator] = h
    nhist[iterator] = n

figure(1)
plot(t, vhist)
title('voltage vs time')

figure(2)
plot(t, mhist, 'y-', t, hhist, 'g.', t, nhist, 'b-')
legend(('m', 'h', 'n'))

figure(3)
gna = gnamax * (mhist ** 3) * hhist
gk = gkmax * nhist ** 4
plot(t, gna, 'r')
plot(t, gk, 'b')
legend(('gna', 'gk'))

show()
