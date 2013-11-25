#!/usr/bin/env python

import sys

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == '__main__':
    if(len(sys.argv) != 2
       or sys.argv[1] not in ['correct_recall', 'incorrect_recall',
                              'spurious_states']
    ):
        print('Incorrect number of parameters.')
        print('Usage: %s correct_recall|incorrect_recall|spurious_states' %
              sys.argv[0])
        sys.exit(1)

    # Get images
    images = sp.loadtxt('images.dat')
    imsize = images[0].size
    imwidth = sp.sqrt(imsize)
    N = 1000
    vecsize = imsize + 5 * N
    #images = sp.where(images >= 0.5, sp.ones(images.shape),
    #                  sp.zeros(images.shape))
    # Vector of neuron values and matrix of weights
    W = sp.zeros((vecsize, vecsize))
    # Initialize matrix via one-shot training
    for i in range(5):
        image = images[i]
        s = image.reshape((imsize, 1))
        u = sp.zeros((vecsize, 1))
        u[:imsize, :] = s
        u[(imsize+N*i):(imsize+N*(i+1)), :] = sp.ones((N, 1))
        # Normalize images while training. We don't want one image to get
        # preference over another just because it has a higher norm.
        # This also makes the matrix unitary, so it doesn't cause values to
        # blow up during the testing phase.
        W += sp.dot(u, u.transpose()) / (5 * la.norm(u)**2)
    print('Training done.')

    # Threshold for equality
    threshold = 1e-3 * sp.ones((vecsize, 1))

    # Look at what is asked
    if sys.argv[1] == 'correct_recall':
        # Randomly pick an image
        index = sp.random.randint(0, 5)
        print 'Expected number: ', index
        v0 = images[index]
        v = sp.zeros(vecsize)
        v[:imsize] = v0
        v[imsize+N*index : imsize+N*(index+1)] = sp.ones(N)
        # Add noise to the image
        noise = sp.random.randint(0, 10, (vecsize,))
        v = sp.where(noise == 1, noise, v)

        # Switch on interactive mode
        plt.ion()
        plt.figure(0)
        v = v.reshape((vecsize, 1))
        plt.imshow(v[:imsize].reshape((imwidth, imwidth)).transpose(), cmap=cm.Greys)
        plt.draw()
        v_prev = v[:, :]
        v = sp.dot(W, v)
        # Keep track of iterations
        i = 0
        # Keep iterating the hopfield network until it stabilizes
        while (abs(v_prev - v) > threshold).any():
            print i
            i += 1
            plt.clf()
            plt.imshow(v[:imsize].reshape((imwidth, imwidth)).transpose(), cmap=cm.Greys)
            plt.draw()
            v_prev[:, :] = v[:, :]
            v = sp.dot(W, v)
        plt.ioff()
        plt.clf()
        plt.imshow(v[:imsize].reshape((imwidth, imwidth)).transpose(), cmap=cm.Greys)
        plt.show()
    if sys.argv[1] == 'incorrect_recall':
        # Randomly pick an image
        index = sp.random.randint(0, 5)
        print 'Expected number: ', index
        v0 = images[index]
        v = sp.zeros(vecsize)
        v[:imsize] = v0
        v[imsize+N*index : imsize+N*(index+1)] = sp.ones(N)
        # Add noise to the image
        noise = sp.random.randint(0, 3, (vecsize,))
        v = sp.where(noise == 1, noise, v)

        # Switch on interactive mode
        plt.ion()
        plt.figure(0)
        v = v.reshape((vecsize, 1))
        plt.imshow(v[:imsize].reshape((imwidth, imwidth)).transpose(), cmap=cm.Greys)
        plt.draw()
        v_prev = v[:, :]
        v = sp.dot(W, v)
        # Keep track of iterations
        i = 0
        # Keep iterating the hopfield network until it stabilizes
        while (abs(v_prev - v) > threshold).any():
            print i
            i += 1
            plt.clf()
            plt.imshow(v[:imsize].reshape((imwidth, imwidth)).transpose(), cmap=cm.Greys)
            plt.draw()
            v_prev[:, :] = v[:, :]
            v = sp.dot(W, v)
        plt.ioff()
        plt.clf()
        plt.imshow(v[:imsize].reshape((imwidth, imwidth)).transpose(), cmap=cm.Greys)
        plt.show()
    if sys.argv[1] == 'spurious_states':
        index = 1
        v0 = images[index]
        v1 = sp.zeros(vecsize)
        v1[:imsize] = v0
        v1[imsize+N*index : imsize+N*(index+1)] = sp.ones(N)
        index = 2
        v0 = images[index]
        v2 = sp.zeros(vecsize)
        v2[:imsize] = v0
        v2[imsize+N*index : imsize+N*(index+1)] = sp.ones(N)

        v = (v1 + v2) / 2

        # Switch on interactive mode
        plt.ion()
        plt.figure(0)
        v = v.reshape((vecsize, 1))
        plt.imshow(v[:imsize].reshape((imwidth, imwidth)).transpose(), cmap=cm.Greys)
        plt.draw()
        v_prev = v[:, :]
        v = sp.dot(W, v)
        # Keep track of iterations
        i = 0
        # Keep iterating the hopfield network until it stabilizes
        while (abs(v_prev - v) > threshold).any():
            print i
            i += 1
            plt.clf()
            plt.imshow(v[:imsize].reshape((imwidth, imwidth)).transpose(), cmap=cm.Greys)
            plt.draw()
            v_prev[:, :] = v[:, :]
            v = sp.dot(W, v)
        plt.ioff()
        plt.clf()
        plt.imshow(v[:imsize].reshape((imwidth, imwidth)).transpose(), cmap=cm.Greys)
        plt.show()
