import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

def plot_functions_1d(len_seq, len_given, len_gen, log_dir, plot_data,
                   hyperparams, h_x_list=None):
  """Plots the predicted mean and variance and the context points.

  Args:
    target_x: An array of shape [B,num_targets,1] that contains the
        x values of the target points.
    target_y: An array of shape [B,num_targets,1] that contains the
        y values of the target points.
    context_x: An array of shape [B,num_contexts,1] that contains
        the x values of the context points.
    context_y: An array of shape [B,num_contexts,1] that contains
        the y values of the context points.
    pred_y: An array of shape [B,num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B,num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
  """
  target_x, target_y, context_x, context_y, pred_y, std = plot_data
  plt.figure(figsize=(6.4, 4.8*(len_seq+len_gen)))
  for t in range(len_seq+len_gen):
      plt.subplot(len_seq+len_gen,1,t+1)
      # Plot everything
      plt.plot(target_x[t][0], target_y[t][0], 'k:', linewidth=2)
      plt.plot(target_x[t][0], pred_y[t][0], 'b', linewidth=2)
      if len(context_x[t]) != 0:
          plt.plot(context_x[t][0], context_y[t][0], 'ko', markersize=10)
      if h_x_list is not None:
          h_y_list = []
          for h_x in h_x_list[t][0]:
            min_val = 10000
            idx = 0
            for i, t_x in enumerate(target_x[t][0]):
                if abs(h_x-t_x) < min_val:
                    min_val = abs(h_x-t_x)
                    idx = i
            h_y_list.append(target_y[t][0][idx])
          plt.plot(h_x_list[t][0],h_y_list, 'ro', markersize=10)
      plt.fill_between(
          target_x[t][0, :, 0],
          pred_y[t][0, :, 0] - std[t][0, :, 0],
          pred_y[t][0, :, 0] + std[t][0, :, 0],
          alpha=0.2,
          facecolor='#65c9f7',
          interpolate=True)

      # Make the plot pretty
      plt.yticks([-4, -2, 0, 2, 4], fontsize=16)
      plt.xticks([-4, -2, 0, 2, 4], fontsize=16)
      #plt.ylim([-2, 2])
      plt.grid('off')
      ax = plt.gca()

  plt.savefig(os.path.join(log_dir,'img.png'))
  plt.close()
  image = misc.imread(os.path.join(log_dir,'img.png'),mode='RGB')

  return image
