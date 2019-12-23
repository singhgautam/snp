import collections
import numpy as np
import tensorflow as tf

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points",
     "hyperparams"))


class GPCurvesReader(object):
  """Generates curves using a Gaussian Process (GP).

  Supports vector inputs (x) and vector outputs (y). Kernel is
  mean-squared exponential, using the x-value l2 coordinate distance scaled
  by some factor chosen randomly in a range. Outputs are
  independent gaussian processes.
  """

  def __init__(self,
               batch_size,
               max_num_context,
               x_size=1,
               y_size=1,
               testing=False,
               len_seq=10,
               len_given=5,
               len_gen=10,
               l1_min=0.7,
               l1_max=1.2,
               l1_vel=0.05,
               sigma_min=1.0,
               sigma_max=1.6,
               sigma_vel=0.05,
               temporal=False,
               case=1,
               ):
    """Creates a regression dataset of functions sampled from a GP.

    Args:
      batch_size: An integer.
      max_num_context: The max number of observations in the context.
      x_size: Integer >= 1 for length of "x values" vector.
      y_size: Integer >= 1 for length of "y values" vector.
      l1_scale: Float; typical scale for kernel distance function.
      sigma_scale: Float; typical scale for variance.
      testing: Boolean that indicates whether we are testing.
               If so there are more targets for visualization.
    """
    self._batch_size = batch_size
    self._max_num_context = max_num_context
    self._x_size = x_size
    self._y_size = y_size
    self._testing = testing
    self._len_seq = len_seq
    self._len_given = len_given
    self._len_gen = len_gen
    self._l1_min = l1_min
    self._l1_max = l1_max
    self._l1_vel = l1_vel
    self._sigma_min = sigma_min
    self._sigma_max = sigma_max
    self._sigma_vel = sigma_vel
    self._temporal = temporal
    self._case = case

    self._noise_factor = 0.1

  def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
    """Applies the Gaussian kernel to generate curve data.

    Args:
      xdata: Tensor of shape [B, num_total_points, x_size] with
          the values of the x-axis data.
      l1: Tensor of shape [B, y_size, x_size], the scale
          parameter of the Gaussian kernel.
      sigma_f: Tensor of shape [B, y_size], the magnitude
          of the std.
      sigma_noise: Float, std of the noise that we add for stability.

    Returns:
      The kernel, a float tensor of shape
      [B, y_size, num_total_points, num_total_points].
    """
    num_total_points = tf.shape(xdata)[1]

    # Expand and take the difference
    xdata1 = tf.expand_dims(xdata, axis=1)# [B, 1, num_total_points, x_size]
    xdata2 = tf.expand_dims(xdata, axis=2)# [B, num_total_points, 1, x_size]
    diff = xdata1 - xdata2# [B, num_total_points, num_total_points, x_size]

    # [B, y_size, num_total_points, num_total_points, x_size]
    norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])

    norm = tf.reduce_sum(
        norm, -1)  # [B, data_size, num_total_points, num_total_points]

    # [B, y_size, num_total_points, num_total_points]
    kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5 * norm)

    # Add some noise to the diagonal to make the cholesky work.
    kernel += (sigma_noise**2) * tf.eye(num_total_points)

    return kernel

  def generate_temporal_curves(self, seed=None):

    # Set kernel parameters
    # Either choose a set of random parameters for the mini-batch
    l1 = tf.random_uniform([self._batch_size, self._y_size,
                            self._x_size], self._l1_min, self._l1_max,
                            seed=seed)
    sigma_f = tf.random_uniform([self._batch_size, self._y_size],
                                self._sigma_min, self._sigma_max, seed=seed)

    l1_vel = tf.random_uniform([self._batch_size, self._y_size,
                               self._x_size],
                                 -1*self._l1_vel, self._l1_vel, seed=seed)
    sigma_f_vel = tf.random_uniform([self._batch_size, self._y_size],
                                 -1*self._sigma_vel, self._sigma_vel,
                                 seed=seed)

    if self._testing:
        num_total_points = 400
    else:
        num_total_points = 100

    y_value_base = tf.random_normal([self._batch_size, self._y_size,
                                     num_total_points, 1],seed=seed)

    curve_list = []
    if (self._case==2) or (self._case==3):
        # sparse time or long term tracking
        idx = tf.random_shuffle(tf.range(self._len_seq),
                                         seed=seed)[:(self._len_given)]
    for t in range(self._len_seq):
        if seed is not None:
            _seed = seed * t
        else:
            _seed = seed
        if self._case==1:    # using len_given
            if t < self._len_given:
                num_context = tf.random_uniform(shape=[], minval=5,
                            maxval=self._max_num_context, dtype=tf.int32,
                            seed=_seed)
            else:
                num_context = tf.constant(0)
                #num_context = tf.constant(1)
        if self._case==2:    # sparse time
            nc_cond = tf.where(tf.equal(idx,t))
            nc_cond = tf.reshape(nc_cond, [-1])
            num_context = tf.cond(tf.equal(tf.size(nc_cond),0),
                          lambda:tf.constant(0),
                          lambda:tf.random_uniform(shape=[], minval=5,
                                            maxval=self._max_num_context,
                                            dtype=tf.int32, seed=_seed))
        if self._case==3:    # long term tracking
            nc_cond = tf.where(tf.equal(idx,t))
            nc_cond = tf.reshape(nc_cond, [-1])
            num_context = tf.cond(tf.equal(tf.size(nc_cond),0),
                          lambda:tf.constant(0),
                          lambda:tf.constant(1))

        if self._temporal:
            encoded_t = None
        else:
            encoded_t = 0.25 + 0.5*t/self._len_seq
        curve_list.append(self.generate_curves(l1, sigma_f, num_context,
                                               y_value_base, _seed,
                                               encoded_t))
        vel_noise = l1_vel * self._noise_factor *  tf. random_normal([
                          self._batch_size,self._y_size, self._x_size],
                          seed=_seed)
        l1 += l1_vel + vel_noise
        vel_noise = sigma_f_vel * self._noise_factor *  tf. random_normal(
                            [self._batch_size, self._x_size], seed=_seed)
        sigma_f += sigma_f_vel + vel_noise

    if self._testing:
        for t in range(self._len_seq,self._len_seq+self._len_gen):
            if seed is not None:
                _seed = seed * t
            else:
                _seed = seed
            num_context = tf.constant(0)

            if self._temporal:
                encoded_t = None
            else:
                encoded_t = 0.25 + 0.5*t/self._len_seq
            curve_list.append(self.generate_curves(l1, sigma_f,
                                               num_context,
                                               y_value_base, _seed,
                                               encoded_t))
            vel_noise = l1_vel * self._noise_factor *  tf. random_normal([
                            self._batch_size,self._y_size, self._x_size],
                            seed=_seed)
            l1 += l1_vel + vel_noise
            vel_noise = sigma_f_vel*self._noise_factor*tf.random_normal(
                            [self._batch_size, self._x_size], seed=_seed)
            sigma_f += sigma_f_vel + vel_noise

    context_x_list, context_y_list = [], []
    target_x_list, target_y_list = [], []
    num_total_points_list = []
    num_context_points_list = []
    for t in range(len(curve_list)):
        (context_x, context_y), target_x = curve_list[t].query
        target_y = curve_list[t].target_y
        num_total_points_list.append(curve_list[t].num_total_points)
        num_context_points_list.append(curve_list[t].num_context_points)
        context_x_list.append(context_x)
        context_y_list.append(context_y)
        target_x_list.append(target_x)
        target_y_list.append(target_y)

    query = ((context_x_list, context_y_list), target_x_list)

    return NPRegressionDescription(
            query=query,
            target_y=target_y_list,
            num_total_points=num_total_points_list,
            num_context_points=num_context_points_list,
            hyperparams=[tf.constant(0)])

  def generate_curves(self, l1, sigma_f, num_context=3,
                      y_value_base=None, seed=None, encoded_t=None):
    """Builds the op delivering the data.

    Generated functions are `float32` with x values between -2 and 2.

    Returns:
      A `CNPRegressionDescription` namedtuple.
    """

    # If we are testing we want to have more targets and have them evenly
    # distributed in order to plot the function.

    if self._testing:
        num_total_points = 400
        num_target = num_total_points
        x_values = tf.tile(
            tf.expand_dims(tf.range(-4., 4., 1. / 50, dtype=tf.float32),
                           axis=0),[self._batch_size, 1])
    else:
        num_total_points = 100
        maxval = self._max_num_context - num_context + 1
        num_target = tf.random_uniform(shape=(), minval=1,
                                       maxval=maxval,
                                       dtype=tf.int32, seed=seed)
        x_values = tf.tile(
            tf.expand_dims(tf.range(-4., 4., 1. / 12.5, dtype=tf.float32),
                           axis=0),[self._batch_size, 1])
    x_values = tf.expand_dims(x_values, axis=-1)
    # During training the number of target points and their x-positions are

    # Pass the x_values through the Gaussian kernel
    # [batch_size, y_size, num_total_points, num_total_points]
    kernel = self._gaussian_kernel(x_values, l1, sigma_f)

    # Calculate Cholesky, using double precision for better stability:
    cholesky = tf.cast(tf.cholesky(tf.cast(kernel, tf.float64)), tf.float32)

    # Sample a curve
    # [batch_size, y_size, num_total_points, 1]
    y_values = tf.matmul(cholesky, y_value_base)

    # [batch_size, num_total_points, y_size]
    y_values = tf.transpose(tf.squeeze(y_values, 3), [0, 2, 1])

    if self._testing:
      # Select the targets
      target_x = x_values
      target_y = y_values

      if encoded_t is not None:
          target_x = tf.concat([
             target_x,
             tf.ones([self._batch_size, num_total_points, 1]) * encoded_t
             ], axis=-1)

      # Select the observations
      idx = tf.random_shuffle(tf.range(num_target), seed=seed)
      context_x = tf.gather(x_values, idx[:num_context], axis=1)
      context_y = tf.gather(y_values, idx[:num_context], axis=1)

      if encoded_t is not None:
          context_x = tf.concat([
             context_x,
             tf.ones([self._batch_size, num_context, 1]) * encoded_t
             ], axis=-1)

    else:
      # Select the targets which will consist of the context points
      # as well as some new target points
      idx = tf.random_shuffle(tf.range(num_total_points), seed=seed)
      target_x = tf.gather(x_values, idx[:num_target + num_context], axis=1)
      target_y = tf.gather(y_values, idx[:num_target + num_context], axis=1)

      if encoded_t is not None:
          target_x = tf.concat([
             target_x,
             tf.ones([self._batch_size, num_target + num_context, 1])
                      * encoded_t], axis=-1)

      # Select the observations
      context_x = tf.gather(x_values, idx[:num_context], axis=1)
      context_y = tf.gather(y_values, idx[:num_context], axis=1)

      if encoded_t is not None:
          context_x = tf.concat([
              context_x,
              tf.ones([self._batch_size, num_context, 1]) * encoded_t
              ], axis=-1)

    query = ((context_x, context_y), target_x)

    return NPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=tf.shape(target_x)[1],
        num_context_points=num_context,
        hyperparams=[tf.constant(0)])
