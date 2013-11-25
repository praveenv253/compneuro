#!/usr/bin/env python

import sys

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == '__main__':
    # Get images
    images = sp.loadtxt('images.dat')
    imsize = images[0].size
    imwidth = sp.sqrt(imsize)
    #images = sp.where(images >= 0.5, sp.ones(images.shape),
    #                  sp.zeros(images.shape))
    # Vector of neuron values and matrix of weights
    v = sp.zeros(imsize)
    W = sp.zeros((imsize, imsize))
    # Initialize matrix via one-shot training
    for image in images:
        s = image.reshape((imsize, 1))
        # Normalize images while training. We don't want one image to get
        # preference over another just because it has a higher norm.
        # This also makes the matrix unitary, so it doesn't cause values to
        # blow up during the testing phase.
        W += sp.dot(s, s.transpose()) / (5 * la.norm(s)**2)
    print('Training done.')

    # Threshold for equality
    threshold = 1e-5 * sp.ones((imsize, 1))

    # Randomly pick an image
    index = sp.random.randint(0, 5)
    v = images[index]
    # Add noise to the image
    noise = sp.random.randint(0, 10, (imsize,))
    v = sp.where(noise == 1, noise, v)

    # Switch on interactive mode
    plt.ion()
    plt.figure(0)
    v = v.reshape((imsize, 1))
    plt.imshow(v.reshape((imwidth, imwidth)).transpose(), cmap=cm.Greys)
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
        plt.imshow(v.reshape((imwidth, imwidth)).transpose(), cmap=cm.Greys)
        plt.draw()
        v_prev[:, :] = v[:, :]
        v = sp.dot(W, v)
    plt.ioff()
    plt.clf()
    plt.imshow(v.reshape((imwidth, imwidth)).transpose(), cmap=cm.Greys)
    plt.show()

