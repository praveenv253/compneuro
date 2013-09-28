#!/usr/bin/env python

"""
This program plots the firing rate of a Hodgkin-Huxley neuron against the
magnitude of injected current (step input).

The regions demarcated by the following thresholds should be found:
1. Threshold current magnitude for the appearance of a single spike
2. Threshold current magnitude for the appearance of continuous firing
3. Threshold current magnitude for the disappearance of continuous firing

The following cut-offs are applied for determination of regions:
1. A "spike" is said to occur if the voltage exceeds 0V
2. The firing is said to be "continuous" if there are more than 10 spikes.
"""

from scipy import *
from matplotlib.pyplot import *

SPIKING_THRESHOLD = 0       # Voltage above which a "spike" is said to occur
CONTINUITY_THRESHOLD = 10   # Number of spikes for "continuous" firing
CURRENT_GRANULARITY = 0.01  # Incremental increase in current step magnitude
MAX_CURRENT = 1             # Maximum current step magnitude

single_spike_threshold = 0
continuous_firing_threshold = 0
distortion_threshold = 0

def calc_voltage_trace(input_current, niter=10000, dt=0.01):
    """Calculates the voltage vs. time for a given input current pulse."""

    gkmax = .36
    vk = -77
    gnamax = 1.20
    vna = 50
    gl = 0.003
    vl = -54.387
    cm = .01

    t = arange(1, niter+1) * dt
    iapp = input_current * ones(niter)

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

    return vhist, niter * dt


def calc_firing_rate(v_trace, tot_time, input_current):
    """Calculates the firing rate of the neuron, given a voltage trace."""

    global single_spike_threshold
    global continuous_firing_threshold
    global distortion_threshold

    # Threshold the trace against the spiking threshold to find out where
    # spikes have occurred.
    thresholded_trace = where( v_trace > SPIKING_THRESHOLD,
                               ones(len(v_trace)), zeros(len(v_trace)) )

    # Count the number of 0->1 transitions to count the number of spikes
    # => Subtract adjacent terms
    transitions = thresholded_trace[1:] - thresholded_trace[:-1]
    # Still has 1->0 transitions => get rid of these
    zero_one_transitions = where(transitions == 1)
    num_spikes = zero_one_transitions[0].size

    # Find exact values of each threshold
    if num_spikes == 1 and not single_spike_threshold:
        single_spike_threshold = input_current
    elif num_spikes >= 10 and not continuous_firing_threshold:
        continuous_firing_threshold = input_current
    elif ( num_spikes < 10 and continuous_firing_threshold != 0
                           and not distortion_threshold ):
        distortion_threshold = input_current

    if num_spikes < CONTINUITY_THRESHOLD:
        return 0
    else:
        return float(num_spikes) / tot_time


if __name__ == '__main__':
    # Input currents
    i_list = arange(0.01, 1, 0.01)
    firing_rate = []
    # Find the firing rate for each input current
    for i in i_list:
        firing_rate.append(calc_firing_rate(*(calc_voltage_trace(i) + (i, ))))
    # Print the thresholds
    print 'Single spike threshold: ', single_spike_threshold, 'uA'
    print 'Appearance of continuous firing: ', continuous_firing_threshold, 'uA'
    print 'Disappearance of continuous firing: ', distortion_threshold, 'uA'
    # Make a plot
    plot(i_list, firing_rate)
    xlabel('Input current ($ \mu A $)')
    ylabel('Firing rate (s)')
    title('Firing rate vs. magnitude of step current')
    show()

