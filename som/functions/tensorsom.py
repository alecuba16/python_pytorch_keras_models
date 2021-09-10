from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys

class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    # To check if the SOM has been trained
    _trained = False

    def __init__(self,m=10, n=10, dim=10, n_iterations=100, alpha=None, sigma=None, random_seed=None):
        # Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            self._alpha = 0.3
        else:
            self._alpha = float(alpha)
        if sigma is None:
            self._sigma = max(m, n) / 2.0
        else:
            self._sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
        # random seed
        if random_seed is not None:
            tf.set_random_seed(random_seed)

    def initialize(self, m, n, dim, n_iterations=100, alpha=None, sigma=None, random_seed=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.

        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """

        # Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            self._alpha = 0.3
        else:
            self._alpha = float(alpha)
        if sigma is None:
            self._sigma = max(m, n) / 2.0
        else:
            self._sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))

        # random seed
        if random_seed is not None:
            tf.set_random_seed(random_seed)

        ##INITIALIZE GRAPH
        self._graph = tf.Graph()

        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE

            # Randomly initialized weightage vectors for all neurons,
            # stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal(
                [m * n, dim]))

            # Matrix of size [m*n, 2] for SOM grid locations
            # of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))

            ##PLACEHOLDERS FOR TRAINING INPUTS
            # We need to assign them as attributes to self, since they
            # will be fed in during training

            # The training vector
            self._vect_input = tf.placeholder("float", [dim])
            # Iteration number
            self._iter_input = tf.placeholder("float")

            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            # Only the final, 'root' training op needs to be assigned as
            # an attribute to self, since all the rest will be executed
            # automatically during training

            # To compute the Best Matching Unit given a vector
            # Basically calculates the Euclidean distance between every
            # neuron's weightage vector and the input, and returns the
            # index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(m * n)])), 2), 1)),
                0)

            # This will extract the location of the BMU based on the BMU's
            # index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2], dtype=np.int64))),
                                 [2])

            # To compute the alpha and sigma values based on iteration
            # number
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.multiply(self._alpha, learning_rate_op)
            _sigma_op = tf.multiply(self._sigma, learning_rate_op)

            # Construct the op that will generate a vector with learning
            # rates for all neurons, based on iteration number and location
            # wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(m * n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            # Finally, the op that will use learning_rate_op to update
            # the weightage vectors of all neurons based on a particular
            # input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                for i in range(m * n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m * n)]),
                       self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)

            ##INITIALIZE SESSION
            self._sess = tf.Session()

            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        # Nested iterations over both dimensions
        # to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
        from datetime import datetime
        from math import floor
        startTime = datetime.now()
        oneIterationTime = None
        # Training iterations
        for iter_no in range(self._n_iterations):
            if iter_no == 0:
                print('Calculating ETA time and %...')
            if iter_no > 0 and oneIterationTime is None:                
                oneIterationTime = (datetime.now() - startTime).total_seconds()
                #total = oneIterationTime * self._n_iterations
                #print(oneIterationTime)
                #print(self._n_iterations-iter_no+1)
            if iter_no > 0:
                eta = (self._n_iterations-iter_no+1)*oneIterationTime
                heta=0
                meta=0
                if eta>3600:
                    heta = floor(eta/3600)
                    eta = eta-(heta*3600)
                if eta>60:
                    meta = floor(eta/60)
                    eta = int(eta-(meta*60))
                print(str(heta)+'h:'+str(meta)+'m:'+str(eta)+'s ,'+str(iter_no*100/self._n_iterations)+' %')
            #if (iter_no%10)==0 :
            #    print(iter_no,end='')
            #else:
            #    print('.', end='')
            sys.stdout.flush()

            # Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})

        # Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
        self._sess.close()
        self._trained = True

    def get_map_size(self):
        return [self._m,self._n]

    def get_trained_model(self):
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return ({'_weightages': self._weightages, '_locations': self._locations, '_centroid_grid': self._centroid_grid, '_trained':self._trained,'_alpha':self._alpha,'_sigma':self._sigma,'_m':self._m,'_n':self._n,'_n_iterations':self._n_iterations})

    def set_trained_model(self,trained_model):
        self._weightages = trained_model['_weightages']
        self._locations = trained_model['_locations']
        self._centroid_grid = trained_model['_centroid_grid']
        self._alpha = trained_model['_alpha']
        self._sigma = trained_model['_sigma']
        self._m = trained_model['_m']
        self._n = trained_model['_n']
        self._n_iterations = trained_model['_n_iterations']
        self._trained = True

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect -
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])

        return to_return

    def fast_norm(self,x):
        """Returns norm-2 of a 1-D numpy array.

        * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
        """
        return np.sqrt(np.dot(x, x.T))

    def distance_map(self):
        """ Returns the distance map of the weights.
            Each cell is the normalised sum of the distances between a neuron and its neighbours.
        """
        weights=self.get_centroids()
        weights=np.array(weights)
        um = np.zeros((weights.shape[0],weights.shape[1]))
        it = np.nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                    if ii >= 0 and ii < weights.shape[0] and jj >= 0 and jj < weights.shape[1]:
                        um[it.multi_index] += self.fast_norm(weights[ii, jj, :]-weights[it.multi_index])
            it.iternext()
        um = um/um.max()
        return um
