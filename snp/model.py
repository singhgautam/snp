import math
import tensorflow as tf
import numpy as np

# utility methods
def batch_mlp(input, output_sizes, variable_scope):

    # Get the shapes of the input and reshape to parallelise across
    # observations
    batch_size, _, filter_size = input.shape.as_list()
    output = tf.reshape(input, (-1, filter_size))
    output.set_shape((None, filter_size))

    # Pass through MLP
    with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
        for i, size in enumerate(output_sizes[:-1]):
            output = tf.nn.relu(
                tf.layers.dense(output, size, name="layer_{}".format(i)))

        # Last layer without a ReLu
        output = tf.layers.dense(
            output, output_sizes[-1], name="layer_{}".format(i + 1))

    # Bring back into original shape
    output = tf.reshape(output, (batch_size, -1, output_sizes[-1]))

    return output

class DeterministicEncoder(object):
  """The Deterministic Encoder."""

  def __init__(self, output_sizes, attention):
    self._output_sizes = output_sizes
    self._attention = attention

  def __call__(self, context_x, context_y, target_x=None, drnn_h=None,
               num_con=None, num_tar=None,
               get_hidden=False, given_hidden=None):

    if given_hidden is None:
        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([context_x, context_y], axis=-1)

        # Pass final axis through MLP
        hidden = batch_mlp(encoder_input, self._output_sizes,
                        "deterministic_encoder")

        # get hidden
        if get_hidden:
            return hidden

    else:
        hidden = given_hidden

    # Apply attention
    with tf.variable_scope("deterministic_encoder", reuse=tf.AUTO_REUSE):
        hidden = self._attention(context_x, target_x, hidden)

    if drnn_h is None:
        hidden = tf.cond(tf.equal(num_con, 0),
                lambda: tf.zeros([target_x.shape[0],
                                num_tar,
                                hidden.shape[-1]]),
                lambda: hidden)
    else:
        drnn_h = tf.tile(tf.expand_dims(drnn_h,axis=1),
                         [1, num_tar, 1])
        hidden = tf.cond(tf.equal(num_con, 0),
                lambda: drnn_h,
                lambda: hidden+drnn_h)

    return hidden

class LatentEncoder(object):
  """The Latent Encoder."""

  def __init__(self, output_sizes, num_latents):

    self._output_sizes = output_sizes
    self._num_latents = num_latents

  def __call__(self, x, y, vrnn_h=None, num=None,
               get_hidden=False, given_hidden=None, irep=None):

    if given_hidden is None:
        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([x, y], axis=-1)

        # Pass final axis through MLP
        hidden = batch_mlp(encoder_input, self._output_sizes,
                        "latent_encoder")

        # only get hidden
        if get_hidden:
            return hidden

    else:
        hidden = given_hidden

    # Aggregator: take the mean over all points
    hidden = tf.reduce_mean(hidden, axis=1)

    # when no data, hidden is equal to 0
    hidden = tf.cond(tf.equal(num, 0),
            lambda: tf.zeros([x.shape[0], self._num_latents]),
            lambda: hidden)

    # temporal or not
    if vrnn_h is not None:
        hidden += vrnn_h

    # Have further MLP layers that map to the parameters
    # of the Gaussian latent
    with tf.variable_scope("latent_encoder", reuse=tf.AUTO_REUSE):
      # First apply intermediate relu layer
      hidden = tf.nn.relu(
          tf.layers.dense(hidden,
                          (self._output_sizes[-1] +
                           self._num_latents)/2,
                          name="penultimate_layer"))
      # Then apply further linear layers to output latent mu
      # and log sigma
      mu = tf.layers.dense(hidden, self._num_latents, name="mean_layer")
      log_sigma = tf.layers.dense(hidden, self._num_latents,
                                  name="std_layer")

    return mu, log_sigma

class Decoder(object):
  """The Decoder."""

  def __init__(self, output_sizes):

    self._output_sizes = output_sizes

  def __call__(self, representation, target_x):

    # concatenate target_x and representation
    hidden = tf.concat([representation, target_x], axis=-1)

    # Pass final axis through MLP
    hidden = batch_mlp(hidden, self._output_sizes, "decoder")

    # Get the mean an the variance
    mu, log_sigma = tf.split(hidden, 2, axis=-1)

    # Bound the variance
    sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

    # Get the distribution
    dist = tf.contrib.distributions.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma)

    return dist, mu, sigma

# Performance optimized version
class LatentModel(object):
    """The (A)NP model."""

    def __init__(self, latent_encoder_output_sizes, num_latents,
                decoder_output_sizes, deterministic_encoder_output_sizes=None,
                attention=None, beta=1.0, dataset='gp'):

        # encoders
        self._latent_encoder = LatentEncoder(latent_encoder_output_sizes,
                                            num_latents)
        self._deterministic_encoder = DeterministicEncoder(
            deterministic_encoder_output_sizes, attention)

        # decoder
        self._decoder = Decoder(decoder_output_sizes)

        self._beta = beta

        # to make seen / unseen plots
        self._index = tf.constant(np.arange(0,2000))

    def __call__(self, query, num_targets, num_contexts, target_y,
                inference=True):

        (context_x, context_y), target_x = query

        # get hidden first
        cont_x = tf.concat(context_x,axis=1)
        cont_y = tf.concat(context_y,axis=1)
        prior_h = self._latent_encoder(cont_x, cont_y, get_hidden=True)
        det_h = self._deterministic_encoder(cont_x, cont_y, get_hidden=True)
        tar_x = tf.concat(target_x,axis=1)
        tar_y = tf.concat(target_y,axis=1)
        post_h = self._latent_encoder(tar_x, tar_y, get_hidden=True)

        mu_list, sigma_list = [], []
        log_p_list, kl_list = [], []
        log_p_seen, log_p_unseen = [], []
        log_p_wo_con, log_p_w_con = 0, 0
        mse_list, mse_wo_con, mse_w_con = [], 0, 0
        cnt_wo, cnt_w = tf.constant(0.0), tf.constant(0.0)
        for t in range(len(context_x)):
            cont_x = tf.concat(context_x[:(t+1)],axis=1)
            cont_y = tf.concat(context_y[:(t+1)],axis=1)
            n_con = np.sum(num_contexts[:(t+1)])
            tar_x = tf.concat(target_x[:(t+1)],axis=1)
            tar_y = tf.concat(target_y[:(t+1)],axis=1)
            n_tar = np.sum(num_targets[:(t+1)])

            #########################################
            # latent encoding
            #########################################
            prior_mu, prior_log_sigma = self._latent_encoder(cont_x, cont_y, num=n_con,
                                            given_hidden=prior_h[:,:n_con])
            prior_sigma = tf.exp(0.5*prior_log_sigma)
            prior_latent_rep = prior_mu + prior_sigma*tf.random_normal(tf.shape(prior_mu),0,1,dtype=tf.float32)

            post_mu, post_log_sigma = self._latent_encoder(tar_x, tar_y, num=n_tar,
                                            given_hidden=post_h[:,:n_tar])
            post_sigma = tf.exp(0.5*post_log_sigma)
            post_latent_rep = post_mu + post_sigma*tf.random_normal(tf.shape(post_mu),0,1,dtype=tf.float32)

            if not inference:
                latent_rep = prior_latent_rep
            else:
                latent_rep = post_latent_rep

            latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1),
                                [1, num_targets[t], 1])

            #########################################
            # det encoding
            #########################################
            deterministic_rep = self._deterministic_encoder(cont_x, cont_y,
                                target_x[t], num_con=n_con,
                                num_tar=num_targets[t],
                                given_hidden=det_h[:,:n_con,:])

            #########################################
            # representation making
            #########################################
            representation = tf.concat([deterministic_rep, latent_rep],
                                        axis=-1)

            #########################################
            # decoding
            #########################################
            dist, mu, sigma = self._decoder(representation, target_x[t])
            mu_list.append(mu)
            sigma_list.append(sigma)

            #########################################
            # calculating loss
            #########################################
            log_p = dist.log_prob(target_y[t])

            kl = -0.5*(-prior_log_sigma+post_log_sigma-(tf.exp(post_log_sigma)+(post_mu-prior_mu)**2) / tf.exp(prior_log_sigma) + 1.0)
            kl = tf.reduce_sum(kl, axis=-1, keepdims=True)

            log_p_seen.append(-1*tf.gather(log_p,
                                        self._index[:num_contexts[t]], axis=1))
            log_p_unseen.append(-1*tf.gather(log_p,
                                    self._index[num_contexts[t]:log_p.shape[1]],
                                    axis=1))
            log_p = -tf.reduce_mean(log_p)
            log_p_list.append(log_p)
            kl_list.append(tf.reduce_mean(kl))
            log_p_wo_con += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:log_p,
                                    lambda:tf.constant(0.0))
            cnt_wo += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(1.0),
                                    lambda:tf.constant(0.0))
            log_p_w_con += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(0.0),
                                    lambda:log_p)
            cnt_w += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(0.0),
                                    lambda:tf.constant(1.0))
            mse = tf.losses.mean_squared_error(target_y[t], mu,
                                        reduction=tf.losses.Reduction.NONE)
            mse = tf.reduce_mean(mse)
            mse_list.append(mse)
            mse_wo_con += tf.cond(tf.equal(num_contexts[t],0), lambda:mse,
                                    lambda:tf.constant(0.0))
            mse_w_con += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(0.0),
                                    lambda:mse)

        #########################################
        # results merging
        #########################################
        mu = mu_list
        sigma = sigma_list

        log_p = np.sum(log_p_list) / len(log_p_list)
        log_p_seen = tf.concat(log_p_seen,axis=-1)
        log_p_unseen = tf.concat(log_p_unseen,axis=-1)
        log_p_seen = tf.reduce_mean(log_p_seen)
        log_p_unseen = tf.reduce_mean(log_p_unseen)
        log_p_w_con = tf.cond(tf.equal(cnt_w,0), lambda:tf.constant(0.0),
                            lambda: log_p_w_con / cnt_w)
        log_p_wo_con = tf.cond(tf.equal(cnt_wo,0), lambda:tf.constant(0.0),
                            lambda: log_p_wo_con / cnt_wo)
        kl = np.sum(kl_list) / len(kl_list)
        loss = log_p + self._beta * kl
        mse = np.sum(mse_list) / len(mse_list)
        mse_w_con = tf.cond(tf.equal(cnt_w,0), lambda:tf.constant(0.0),
                            lambda: mse_w_con / cnt_w)
        mse_wo_con = tf.cond(tf.equal(cnt_wo,0), lambda:tf.constant(0.0),
                            lambda: mse_wo_con / cnt_wo)

        debug_metrics = (log_p, log_p_list, log_p_w_con, log_p_wo_con,
                        mse, mse_list, mse_w_con, mse_wo_con, log_p_seen,
                        log_p_unseen)

        return mu, sigma, log_p, kl, loss, debug_metrics

class TemporalLatentModel(object):
    """The SNP model."""

    def __init__(self, latent_encoder_output_sizes, num_latents,
                decoder_output_sizes, deterministic_encoder_output_sizes=None,
                attention=None, beta=1.0, alpha=0.0, dataset='gp'):

        # encoders
        self._latent_encoder = LatentEncoder(latent_encoder_output_sizes,
                                            num_latents)
        self._deterministic_encoder = DeterministicEncoder(
            deterministic_encoder_output_sizes, attention)
        # decoder
        self._decoder = Decoder(decoder_output_sizes)

        # rnn modules
        self._drnn = tf.contrib.rnn.BasicLSTMCell(num_latents)
        self._vrnn = tf.contrib.rnn.BasicLSTMCell(num_latents)

        self._beta = beta

        # to make unseen/seen plots
        self._index = tf.constant(np.arange(0,2000))

    def __call__(self, query, num_targets, num_contexts, target_y,
                inference=True):

        (context_x, context_y), target_x = query

        len_seq = len(context_x)

        batch_size = context_x[0].shape[0]

        # latent rnn state initialization
        init_vrnn_state = self._vrnn.zero_state(batch_size, dtype=tf.float32)
        init_vrnn_hidden = init_vrnn_state[1]
        vrnn_state = init_vrnn_state
        vrnn_hidden = init_vrnn_hidden
        latent_rep = init_vrnn_hidden

        # det rnn state initialization
        init_drnn_state = self._drnn.zero_state(batch_size,
                                                dtype=tf.float32)
        init_drnn_hidden = init_drnn_state[1]
        drnn_state = init_drnn_state
        drnn_hidden = init_drnn_hidden
        avg_det_rep = init_drnn_hidden

        mu_list, sigma_list = [], []
        log_p_list, kl_list = [], []
        log_p_seen, log_p_unseen = [], []
        log_p_wo_con, log_p_w_con = 0, 0
        mse_list, mse_wo_con, mse_w_con = [], 0, 0
        cnt_wo, cnt_w = tf.constant(0.0), tf.constant(0.0)
        for t in range(len_seq):

            # current observations
            cont_x = context_x[t]
            cont_y = context_y[t]
            tar_x = target_x[t]

            ########################################
            # latent encoding
            ########################################
            lat_en_c = self._latent_encoder(cont_x, cont_y, get_hidden=True)
            lat_en_c_mean = tf.reduce_mean(lat_en_c, axis=1)

            latent_rep = tf.cond(tf.equal(num_contexts[t],0),
                                lambda: latent_rep,
                                lambda: latent_rep + lat_en_c_mean)

            vrnn_hidden, vrnn_state = self._vrnn(latent_rep, vrnn_state)

            prior_mu, prior_log_sigma = self._latent_encoder(cont_x, cont_y, vrnn_hidden,
                                            num_contexts[t], given_hidden=lat_en_c)
            prior_sigma = tf.exp(0.5*prior_log_sigma)
            tar_y = target_y[t]
            post_mu, post_log_sigma = self._latent_encoder(tar_x, tar_y, vrnn_hidden,
                                                num_targets[t])
            post_sigma = tf.exp(0.5*post_log_sigma)
            prior_latent_rep = prior_mu + prior_sigma*tf.random_normal(tf.shape(prior_mu),0,1,dtype=tf.float32)
            post_latent_rep = post_mu + post_sigma*tf.random_normal(tf.shape(post_mu),0,1,dtype=tf.float32)
            if not inference:
                latent_rep = prior_latent_rep
            else:
                latent_rep = post_latent_rep

            latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1),
                                [1, num_targets[t], 1])

            ########################################
            # det encoding
            ########################################
            det_en_c = self._deterministic_encoder(cont_x, cont_y, get_hidden=True)
            det_en_c_mean = tf.reduce_mean(det_en_c, axis=1)

            avg_det_rep = tf.cond(tf.equal(num_contexts[t],0),
                                lambda: avg_det_rep,
                                lambda: avg_det_rep + det_en_c_mean)
            drnn_hidden, drnn_state = self._drnn(avg_det_rep, drnn_state)

            deterministic_rep = self._deterministic_encoder(cont_x,
                                                            cont_y,
                                                            tar_x,
                                                            drnn_hidden,
                                                            num_con=num_contexts[t],
                                                            num_tar=num_targets[t],
                                                            given_hidden=det_en_c)
            avg_det_rep = tf.reduce_mean(deterministic_rep, axis=1)

            ########################################
            # representation merging
            ########################################
            representation = tf.concat([deterministic_rep, latent_rep], axis=-1)
            latent_rep = tf.reduce_mean(latent_rep, axis=1)

            ########################################
            # decoding
            ########################################
            dist, mu, sigma = self._decoder(representation, tar_x)
            mu_list.append(mu)
            sigma_list.append(sigma)

            ########################################
            # calculating loss
            ########################################
            log_p = dist.log_prob(tar_y)
            log_p_seen.append(-1*tf.gather(log_p,
                                        self._index[:num_contexts[t]], axis=1))
            log_p_unseen.append(-1*tf.gather(log_p,
                                        self._index[num_contexts[t]:log_p.shape[1]],
                                        axis=1))
            kl = -0.5*(-prior_log_sigma+post_log_sigma-(tf.exp(post_log_sigma)+(post_mu-prior_mu)**2) / tf.exp(prior_log_sigma) + 1.0)
            kl = tf.reduce_sum(kl, axis=-1, keepdims=True)
            log_p = -tf.reduce_mean(log_p)
            log_p_list.append(log_p)
            kl_list.append(tf.reduce_mean(kl))
            log_p_wo_con += tf.cond(tf.equal(num_contexts[t],0), lambda:log_p,
                                    lambda:tf.constant(0.0))
            cnt_wo += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(1.0),
                                    lambda:tf.constant(0.0))
            log_p_w_con += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(0.0),
                                    lambda:log_p)
            cnt_w += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(0.0),
                                    lambda:tf.constant(1.0))
            mse = tf.losses.mean_squared_error(target_y[t], mu,
                                        reduction=tf.losses.Reduction.NONE)
            mse = tf.reduce_mean(mse)
            mse_list.append(mse)
            mse_wo_con += tf.cond(tf.equal(num_contexts[t],0), lambda:mse,
                                    lambda:tf.constant(0.0))
            mse_w_con += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(0.0),
                                    lambda:mse)

        ########################################
        # result merging
        ########################################
        mu = mu_list
        sigma = sigma_list

        log_p = np.sum(log_p_list) / len(log_p_list)
        log_p_seen = tf.concat(log_p_seen,axis=-1)
        log_p_unseen = tf.concat(log_p_unseen,axis=-1)
        log_p_seen = tf.reduce_mean(log_p_seen)
        log_p_unseen = tf.reduce_mean(log_p_unseen)
        log_p_w_con = tf.cond(tf.equal(cnt_w,0), lambda:tf.constant(0.0),
                            lambda: log_p_w_con / cnt_w)
        log_p_wo_con = tf.cond(tf.equal(cnt_wo,0), lambda:tf.constant(0.0),
                            lambda: log_p_wo_con / cnt_wo)
        kl = np.sum(kl_list) / len(kl_list)
        log_p = log_p
        kl = kl
        loss = log_p + self._beta * kl

        mse = np.sum(mse_list) / len(mse_list)
        mse_w_con = tf.cond(tf.equal(cnt_w,0), lambda:tf.constant(0.0),
                            lambda: mse_w_con / cnt_w)
        mse_wo_con = tf.cond(tf.equal(cnt_wo,0), lambda:tf.constant(0.0),
                            lambda: mse_wo_con / cnt_wo)

        debug_metrics = (log_p, log_p_list, log_p_w_con, log_p_wo_con,
                        mse, mse_list, mse_w_con, mse_wo_con, log_p_seen,
                        log_p_unseen)

        return mu, sigma, log_p, kl, loss, debug_metrics

def uniform_attention(q, v):
  """Uniform attention. Equivalent to np.

  Args:
    q: queries. tensor of shape [B,m,d_k].
    v: values. tensor of shape [B,n,d_v].

  Returns:
    tensor of shape [B,m,d_v].
  """
  total_points = tf.shape(q)[1]
  rep = tf.reduce_mean(v, axis=1, keepdims=True)  # [B,1,d_v]
  rep = tf.tile(rep, [1, total_points, 1])
  return rep

def laplace_attention(q, k, v, scale, normalise):
  """Computes laplace exponential attention.

  Args:
    q: queries. tensor of shape [B,m,d_k].
    k: keys. tensor of shape [B,n,d_k].
    v: values. tensor of shape [B,n,d_v].
    scale: float that scales the L1 distance.
    normalise: Boolean that determines whether weights sum to 1.

  Returns:
    tensor of shape [B,m,d_v].
  """

  k = tf.expand_dims(k, axis=1)  # [B,1,n,d_k]
  q = tf.expand_dims(q, axis=2)  # [B,m,1,d_k]
  unnorm_weights = - tf.abs((k - q) / scale)  # [B,m,n,d_k]
  unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)  # [B,m,n]
  if normalise:
    weight_fn = tf.nn.softmax
  else:
    weight_fn = lambda x: 1 + tf.tanh(x)
  weights = weight_fn(unnorm_weights)  # [B,m,n]
  rep = tf.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
  return rep


def dot_product_attention(q, k, v, normalise):
  """Computes dot product attention.

  Args:
    q: queries. tensor of  shape [B,m,d_k].
    k: keys. tensor of shape [B,n,d_k].
    v: values. tensor of shape [B,n,d_v].
    normalise: Boolean that determines whether weights sum to 1.

  Returns:
    tensor of shape [B,m,d_v].
  """
  d_k = tf.shape(q)[-1]
  scale = tf.sqrt(tf.cast(d_k, tf.float32))
  unnorm_weights = tf.einsum('bjk,bik->bij', k, q) / scale  # [B,m,n]
  if normalise:
    weight_fn = tf.nn.softmax
  else:
    weight_fn = tf.sigmoid
  weights = weight_fn(unnorm_weights)  # [B,m,n]
  rep = tf.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
  return rep


def multihead_attention(q, k, v, num_heads=8):
  """Computes multi-head attention.

  Args:
    q: queries. tensor of  shape [B,m,d_k].
    k: keys. tensor of shape [B,n,d_k].
    v: values. tensor of shape [B,n,d_v].
    num_heads: number of heads. Should divide d_v.

  Returns:
    tensor of shape [B,m,d_v].
  """
  d_k = q.get_shape().as_list()[-1]
  d_v = v.get_shape().as_list()[-1]
  head_size = d_v / num_heads
  key_initializer = tf.random_normal_initializer(stddev=d_k**-0.5)
  value_initializer = tf.random_normal_initializer(stddev=d_v**-0.5)
  rep = tf.constant(0.0)
  for h in range(num_heads):
    o = dot_product_attention(
        tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                   name='wq%d' % h, use_bias=False, padding='VALID')(q),
        tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                   name='wk%d' % h, use_bias=False, padding='VALID')(k),
        tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                   name='wv%d' % h, use_bias=False, padding='VALID')(v),
        normalise=True)
    rep += tf.layers.Conv1D(d_v, 1, kernel_initializer=value_initializer,
                      name='wo%d' % h, use_bias=False, padding='VALID')(o)
  return rep

class Attention(object):
  """The Attention module."""

  def __init__(self, rep, output_sizes, att_type, scale=1., normalise=True,
               num_heads=8):
    """Create attention module.

    Takes in context inputs, target inputs and
    representations of each context input/output pair
    to output an aggregated representation of the context data.
    Args:
      rep: transformation to apply to contexts before computing attention.
          One of: ['identity','mlp'].
      output_sizes: list of number of hidden units per layer of mlp.
          Used only if rep == 'mlp'.
      att_type: type of attention. One of the following:
          ['uniform','laplace','dot_product','multihead']
      scale: scale of attention.
      normalise: Boolean determining whether to:
          1. apply softmax to weights so that they sum to 1 across context pts or
          2. apply custom transformation to have weights in [0,1].
      num_heads: number of heads for multihead.
    """
    self._rep = rep
    self._output_sizes = output_sizes
    self._type = att_type
    self._scale = scale
    self._normalise = normalise
    if self._type == 'multihead':
      self._num_heads = num_heads

  def __call__(self, x1, x2, r):
    """Apply attention to create aggregated representation of r.

    Args:
      x1: tensor of shape [B,n1,d_x].
      x2: tensor of shape [B,n2,d_x].
      r: tensor of shape [B,n1,d].

    Returns:
      tensor of shape [B,n2,d]

    Raises:
      NameError: The argument for rep/type was invalid.
    """
    if self._rep == 'identity':
      k, q = (x1, x2)
    elif self._rep == 'mlp':
      # Pass through MLP
      k = batch_mlp(x1, self._output_sizes, "attention")
      q = batch_mlp(x2, self._output_sizes, "attention")
    else:
      raise NameError("'rep' not among ['identity','mlp']")

    if self._type == 'uniform':
      rep = uniform_attention(q, r)
    elif self._type == 'laplace':
      rep = laplace_attention(q, k, r, self._scale, self._normalise)
    elif self._type == 'dot_product':
      rep = dot_product_attention(q, k, r, self._normalise)
    elif self._type == 'multihead':
      rep = multihead_attention(q, k, r, self._num_heads)
    else:
      raise NameError(("'att_type' not among ['uniform','laplace','dot_product'"
                       ",'multihead']"))

    return rep
