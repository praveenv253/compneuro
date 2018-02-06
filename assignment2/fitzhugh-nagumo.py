#!/usr/bin/env python3

"""
Program to simulate the Fitzhugh-Nagumo model of the neuron for different
values of input current, and find thresholds of excitability, oscillation and
bistability.
"""

# Fitzhugh-Nagumo equations:
#           dv/dt = gamma*v - v^3 / 3 - w + i
#           dw/dt = (v + alpha - beta*w) / tau

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


def fwd_euler_step(v0, w0, dt, i_app, alpha, beta, gamma, tau):
    """
    Performs a single forward Euler step in simulating the FN model.
    Returns the new value of v and w.
    """

    # Discrete-time representation of the differential equations
    v = v0 + dt * (gamma*v0 - v0**3/3 - w0 + i_app)
    w = w0 + dt * (v0 + alpha - beta*w0) / tau

    return v, w


def fwd_euler(v0, w0, dt, i_app, t, alpha, beta, gamma, tau, plot=True):
    """
    Performs a forward Euler computation to determine the time-trace of voltage
    for a given amount of time `t` in time steps of `dt`.
    Returns the time trace.
    """

    # Initialize phase-space parameters
    n = i_app.size
    v = sp.empty(n)
    w = sp.empty(n)
    v[0] = v0
    w[0] = w0

    # Compute nullclines for plotting phase-space evolution
    v_axis = sp.linspace(-2.5, 2.5, 1000)
    w_nullcline = (v_axis + alpha) / beta
    v_nullcline = gamma * v_axis - v_axis**3 / 3 + i_app[0]

    if plot:
        # Set figure and subplot properties
        plt.ion()
        fig = plt.figure(0, figsize=(12, 9))
        phase_diagram = fig.add_subplot(311)
        phase_diagram.set_title('Phase plane')
        phase_diagram.set_xlabel(r'$v$')
        phase_diagram.set_ylabel(r'$w$')
        [vnc,] = phase_diagram.plot(v_axis, v_nullcline, 'C0-')
        [wnc,] = phase_diagram.plot(v_axis, w_nullcline, 'C1-')
        [pos,] = phase_diagram.plot(0, 0, 'C2o')
        vtrace = fig.add_subplot(312)
        vtrace.set_title('Voltage trace')
        vtrace.set_xlabel(r'$t$')
        vtrace.set_ylabel(r'$v$')
        itrace = fig.add_subplot(313)
        itrace.set_title('External current')
        itrace.set_xlabel(r'$t$')
        itrace.set_ylabel(r'$I_{app}$')
        fig.tight_layout(pad=2)

    for i in range(1, n):
        # Increment time step
        v[i], w[i] = fwd_euler_step(v[i-1], w[i-1], dt, i_app[i], alpha, beta,
                                    gamma, tau)
        # Re-compute nullcline in case exciting current has changed
        if i_app[i] != i_app[i-1]:
            v_nullcline += i_app[i] - i_app[i-1]
        # Plot the phase plane with the nullclines and present position
        if plot:
            # Plot the phase diagram
            phase_diagram.lines.remove(vnc)
            phase_diagram.lines.remove(pos)
            [vnc,] = phase_diagram.plot(v_axis, v_nullcline, 'C0-')
            phase_diagram.plot([v[i-1], v[i]], [w[i-1], w[i]], 'k-')
            [pos,] = phase_diagram.plot(v[i], w[i], 'C2o')
            # Plot the v-trace
            vtrace.plot([(i-1)*dt, i*dt], [v[i-1], v[i]], 'C0-')
            # Plot the i-trace
            itrace.plot([(i-1)*dt, i*dt], [i_app[i-1], i_app[i]], 'C0-')
            # Redraw the figure
            plt.draw()

    # Show the final plot
    if plot:
        plt.ioff()
        plt.show()

    # Return the voltage trace
    return v


def excitability():
    """Show excitability of the FN-neuron."""

    v0 = -1.5     # Initial voltage
    w0 = -0.5     # Initial w-parameter
    dt = 0.3      # Simulation time step
    t = 100       # Total simulation time
    n = int(t / dt)

    # FN model parameters
    alpha = 0.7
    beta = 0.8
    gamma = 1
    tau = 1 / 0.08
    i_app = -0.5 * sp.ones(n)  # Applied current
    i_app[0.15*n:0.2*n] = -0.3333 * sp.ones(0.05*n + 1)
    i_app[0.4*n:0.45*n] = -0.1667 * sp.ones(0.05*n)
    i_app[0.65*n:0.7*n] = 0 * sp.ones(0.05*n + 1)

    v_t = fwd_euler(v0, w0, dt, i_app, t, alpha, beta, gamma, tau)


def spiking():
    """Show spiking in the FN-neuron."""

    v0 = -0.7     # Initial voltage
    w0 = -0.5     # Initial w-parameter
    dt = 0.3      # Simulation time step
    t = 100       # Total simulation time

    # FN model parameters
    alpha = 0.7
    beta = 0.8
    gamma = 1
    tau = 1 / 0.08
    n = int(t / dt)
    i_app = 0.85 * sp.ones(n)  # Applied current

    v_t = fwd_euler(v0, w0, dt, i_app, t, alpha, beta, gamma, tau)


def depolarization():
    """Show depolarization in the FN-neuron."""

    v0 = -0.7     # Initial voltage
    w0 = -0.5     # Initial w-parameter
    dt = 0.3      # Simulation time step
    t = 100       # Total simulation time
    n = int(t / dt)

    # FN model parameters
    alpha = 0.7
    beta = 0.8
    gamma = 1
    tau = 1 / 0.08
    i_app = 2 * sp.ones(n)  # Applied current

    v_t = fwd_euler(v0, w0, dt, i_app, t, alpha, beta, gamma, tau)


def bistability():
    """Show bistability in the FN-neuron."""

    v0 = -2       # Initial voltage
    w0 = -1       # Initial w-parameter
    i_app = 0.85  # Applied current
    dt = 0.3      # Simulation time step
    t = 100       # Total simulation time

    # FN model parameters
    alpha = 0.7
    beta = 0.8
    gamma = 2
    tau = 1 / 0.08
    n = int(t / dt)
    i_app = 0.85 * sp.ones(n)  # Applied current
    impulse_duration = int(0.03*n)
    i_app[0.2*n:0.2*n+impulse_duration] = 1 * sp.ones(impulse_duration)
    i_app[0.7*n:0.7*n+impulse_duration] = 0.7 * sp.ones(impulse_duration)

    v_t = fwd_euler(v0, w0, dt, i_app, t, alpha, beta, gamma, tau)


def calc_firing_rate(v_trace, tot_time):
    """Calculates the firing rate of the neuron, given a voltage trace."""

    # "Spiking" happens only when at least 2 spikes are produced.
    CONTINUITY_THRESHOLD = 2
    SPIKING_THRESHOLD = 2

    # Threshold the trace against the spiking threshold to find out where
    # spikes have occurred.
    thresholded_trace = sp.where(v_trace > SPIKING_THRESHOLD,
                                 sp.ones(len(v_trace)), sp.zeros(len(v_trace)))

    # Count the number of 0->1 transitions to count the number of spikes
    # => Subtract adjacent terms
    transitions = thresholded_trace[1:] - thresholded_trace[:-1]
    # Still has 1->0 transitions => get rid of these
    zero_one_transitions = sp.where(transitions == 1)
    num_spikes = zero_one_transitions[0].size

    if num_spikes < CONTINUITY_THRESHOLD:
        return 0
    else:
        return float(num_spikes) / tot_time


def thresholds():
    """Print excitability and depolarization thresholds of the FN-neuron."""

    v0 = -0.7     # Initial voltage
    w0 = -0.5     # Initial w-parameter
    dt = 0.3      # Simulation time step
    t = 100       # Total simulation time
    n = int(t / dt)

    # FN model parameters
    alpha = 0.7
    beta = 0.8
    gamma = 1
    tau = 1 / 0.08
    # List of currents to check for thresholds
    i_base = -0.5
    currents = sp.linspace(i_base, 1.5, 100)

    excitability_threshold = None
    depolarization_threshold = None

    for i in currents:
        i_app = i * sp.ones(n)  # Applied current
        # Find the voltage trace for this value of input current.
        v_t = fwd_euler(v0, w0, dt, i_app, t, alpha, beta, gamma, tau, False)
        # Calculate the firing rate for this voltage trace.
        f = calc_firing_rate(v_t, t)
        # Set the excitability threshold the first time spiking appears.
        if f and excitability_threshold is None:
            excitability_threshold = i - i_base
        # Set the depolarization threshold the first time spiking ceases.
        # (i.e, spiking has appeared, but no longer appears now)
        if( f == 0 and excitability_threshold is not None
                   and depolarization_threshold is None ):
            depolarization_threshold = i - i_base

    print(( 'Excitability threshold (over base current): '
           + str(excitability_threshold) ))
    print(( 'Depolarization threshold (over base current): '
           + str(depolarization_threshold) ))

    return excitability_threshold, depolarization_threshold


def usage(prog_name):
    """Print help."""
    print('Usage:')
    print(('    ' + str(prog_name) + ' --excitability | --spiking |'
                                    ' --bistability | --depolarization |'
                                    ' --thresholds'))


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        usage(sys.argv[0])
        sys.exit(1)
    if sys.argv[1] == '--excitability':
        excitability()
    elif sys.argv[1] == '--spiking':
        spiking()
    elif sys.argv[1] == '--bistability':
        bistability()
    elif sys.argv[1] == '--depolarization':
        depolarization()
    elif sys.argv[1] == '--thresholds':
        thresholds()
    else:
        usage(sys.argv[0])
