#!/usr/bin/env python

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

    v = v0 + dt * (gamma*v0 - v0**3/3 - w0 + i_app)
    w = w0 + dt * (v0 + alpha - beta*w0) / tau

    return v, w

def fwd_euler(v0, w0, dt, i_app, t, alpha, beta, gamma, tau):
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

    # Set figure and subplot properties
    plt.ion()
    plt.figure(0, figsize=(12, 9))
    plt.subplot(311)
    plt.subplot(312)
    plt.title('Voltage trace')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$v$')
    plt.subplot(313)
    plt.title('External current')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$I_{app}$')
    plt.tight_layout(pad=2)

    for i in xrange(1, n):
        # Increment time step
        v[i], w[i] = fwd_euler_step(v[i-1], w[i-1], dt, i_app[i], alpha, beta,
                                    gamma, tau)
        # Re-compute nullcline in case exciting current has changed
        if i_app[i] != i_app[i-1]:
            v_nullcline += i_app[i] - i_app[i-1]
        # Clear the figure
        # Plot the phase plane with the nullclines and present position
        plt.subplot(311)
        plt.cla()
        plt.title('Phase plane')
        plt.xlabel(r'$v$')
        plt.ylabel(r'$w$')
        plt.plot(v_axis, v_nullcline, 'b-')
        plt.plot(v_axis, w_nullcline, 'g-')
        plt.plot(v[i], w[i], 'ro')
        # Plot the v-trace
        plt.subplot(312)
        plt.plot([(i-1)*dt, i*dt], [v[i-1], v[i]], 'b-')
        # Plot the i-trace
        plt.subplot(313)
        plt.plot([(i-1)*dt, i*dt], [i_app[i-1], i_app[i]], 'b-')
        # Redraw the figure
        plt.draw()

    # Return the voltage trace
    return v

def excitability():
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
    i_app = -0.5 * sp.ones(n)  # Applied current
    i_app[0.2*n:0.3*n] = 1 * sp.ones(0.1*n)

    v_t = fwd_euler(v0, w0, dt, i_app, t, alpha, beta, gamma, tau)

def spiking():
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

def bistability():
    v0 = -2     # Initial voltage
    w0 = -1     # Initial w-parameter
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
    impulse_duration = int(0.0275*n)
    i_app[0.2*n:0.2*n+impulse_duration] = 1 * sp.ones(impulse_duration)

    v_t = fwd_euler(v0, w0, dt, i_app, t, alpha, beta, gamma, tau)

def usage(prog_name):
    print('Usage:')
    print('    ' + str(prog_name) + ' [ --excitability | --spiking '
                                    '| --bistability ]')

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
    else:
        usage(sys.argv[0])
