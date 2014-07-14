#!/usr/bin/env python

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

NUM_HIDDEN_NEURONS = 10
ETA = 0.5
ERROR_THRESHOLD = 0.5

def g(x, lamda=0.5):
    return 1 / (1 + sp.exp(- lamda * x))

def gprime(x, lamda=0.5):
    return lamda * sp.exp(- lamda * x) / (1 + sp.exp(-lamda * x)) ** 2

class MLP(object):
    def __init__(self, num_inputs, num_hidden_neurons, num_outputs):
        self.nx = num_inputs + 1               # +1 for 'b'
        self.nv = num_hidden_neurons
        self.ny = num_outputs
        self.wf = sp.zeros((self.nv, self.nx))
        self.ws = sp.zeros((self.ny, self.nv))

    def train(self, inputs, outputs, eta=0.1):
        avg_errors = []
        t = 0
        num_images = inputs.shape[0]
        inputs = sp.hstack((inputs, -1 * sp.ones((num_images, 1))))  # -1 for b
        while(1):
            print 'Epoch %d' % t
            t += 1
            i = 0
            errors = []
            for inpt in inputs:
                x = inpt.reshape((inpt.size, 1))
                d = outputs[i]
                d = d.reshape((d.size, 1))
                hf = sp.dot(self.wf, x)
                v = g(hf)
                hs = sp.dot(self.ws, v)
                y = g(hs)
                error = d - y
                delta_s = sp.transpose(error * gprime(hs))        # Row vector
                delta_f = sp.dot(delta_s, self.ws) * gprime(hf.transpose())
                delta_ws = eta * sp.dot(v, delta_s).transpose()
                delta_wf = eta * sp.dot(x, delta_f).transpose()
                self.ws += delta_ws
                self.wf += delta_wf
                errors.append(la.norm(error))
                i += 1
            errors = sp.array(errors)
            avg_error = errors.mean()
            avg_errors.append(avg_error)
            a = sp.array(avg_errors)
            print 'Average error: %f' % avg_error
            if avg_error < ERROR_THRESHOLD or (
                    t > 5 and (abs(a[-5:]-a[-1]) < 1e-8).all()
            ):
                break
            if t > 2 and a[-1] > a[-2]:
                eta *= 2.0 / 3
                print 'Eta changed: %f' % eta
        return avg_errors

    def test(self, inputs, outputs):
        error = 0.0
        i = 0
        num_images = inputs.shape[0]
        inputs = sp.hstack((inputs, -1 * sp.ones((num_images, 1))))  # -1 for b
        for inpt in inputs:
            x = inpt.reshape((inpt.size, 1))
            d = outputs[i]
            d = d.reshape((d.size, 1))
            hf = sp.dot(self.wf, x)
            v = g(hf)
            hs = sp.dot(self.ws, v)
            y = g(hs)
            index = sp.argmax(y)
            if d[index] != 1:
                error += 1
            i += 1
        print 'Percentage correct: %f' % (100 - 100 * error / i)
        return (error / i)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 3:
        print 'Too many parameters'
        sys.exit()
    else:
        if len(sys.argv) == 3:
            ETA = float(sys.argv[2])
            NUM_HIDDEN_NEURONS = int(sys.argv[1])
        elif len(sys.argv) == 2:
            NUM_HIDDEN_NEURONS = int(sys.argv[1])

    images = sp.loadtxt('images1000.dat')
    labels = sp.loadtxt('labels1000.dat')
    test_images = sp.loadtxt('test_images.dat')
    test_labels = sp.loadtxt('test_labels.dat')

    mlp = MLP(784, NUM_HIDDEN_NEURONS, 10)
    errors = mlp.train(images, labels, ETA)
    plt.plot(range(len(errors)), errors, 'b-')
    plt.xlabel('Epochs')
    plt.ylabel(r'Average error ($||\bf{y}-\bf{d}||$)')
    mlp.test(test_images, test_labels)
    plt.show()

