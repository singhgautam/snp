import os, copy
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def create_directory(directory):
    for i in range(len(directory.split('/'))):
        if directory.split('/')[i] != '':
            sub_dic ='/'.join(directory.split('/')[:(i+1)])
            if not os.path.exists(sub_dic):
                os.makedirs(sub_dic)

def reordering(whole_query, target_y, pred_y, std_y, temporal=False):

    (context_x, context_y), target_x = whole_query

    if not temporal:
        for i in range(len(context_x)):
            context_x[i] = context_x[i][:,:,:-1]
        target_x = np.array(target_x)[:,:,:,:-1]

    context_x_list = context_x
    context_y_list = context_y
    target_x_list = target_x
    target_y_list = target_y
    pred_y_list = pred_y
    std_y_list = std_y

    return (target_x_list, target_y_list, context_x_list, context_y_list,
           pred_y_list, std_y_list)

def get_param_names(mu, sigma, query, ntp, ncp, target_y):

    param_names = {}

    param_names['mu'] = [item.name for item in mu]
    param_names['sigma'] = [item.name for item in sigma]
    param_names['target_y'] = [item.name for item in target_y]
    (context_x, context_y), target_x = query
    param_names['context_x'] = [item.name for item in context_x]
    param_names['context_y'] = [item.name for item in context_y]
    param_names['target_x'] = [item.name for item in target_x]
    param_names['num_total_points'] = [item.name for item in ntp]
    param_names['num_context_points'] = [item.name for item in ncp]

    return param_names

def making_feed_dict(data, query, ntp, ncp, tar_y):

    feed_dict = {}

    (dcon_x, dcon_y), dtar_x = data.query
    dtar_y = data.target_y
    dntp = data.num_total_points
    dncp = data.num_context_points

    (con_x, con_y), tar_x = query

    for t in range(len(con_x)):

        feed_dict[con_x[t]] = dcon_x[t]
        feed_dict[con_y[t]] = dcon_y[t]
        feed_dict[tar_x[t]] = dtar_x[t]
        feed_dict[tar_y[t]] = dtar_y[t]
        feed_dict[ntp[t]] = dntp[t]
        feed_dict[ncp[t]] = dncp[t]

    return feed_dict
