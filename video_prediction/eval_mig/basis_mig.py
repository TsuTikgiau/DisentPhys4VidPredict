# This file is derived from Ricky Tian Qi Chen's beta-TCVAE project
# (https://github.com/rtqichen/beta-tcvae) with the MIT license.
# The license can be found in LICENSE.md file under the root.


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf

import video_prediction.utils.math as umath


class EvalMigBasis(object):

    def __init__(self, model, dataset, batch_size):
        inputs, inputs_init_op = dataset.make_batch(batch_size, reinit=True)
        input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}

        # setting is not needed in the model
        for key in list(input_phs.keys()):
            if 'setting' in key:
                _ = input_phs.pop(key)

        with tf.variable_scope(''):
            # !!! NEED TO BE MODIFIED ACCORDING TO THE IMPLEMENTATION OF YOUR MODEL
            model.build_graph(input_phs)  # build the graph of your model
            # !!! NEED TO BE MODIFIED ACCORDING TO THE IMPLEMENTATION OF YOUR MODEL end

        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        self.sess = sess
        self.model = model
        self.inputs = inputs
        self.inputs_init = inputs_init_op
        self.input_phs = input_phs

        # !!! NEED TO BE MODIFIED ACCORDING TO THE IMPLEMENTATION OF YOUR MODEL
        self.latent_dim = self.model.outputs['q_samples'].shape[-1].value  # the number of dimensions of latent code
        self.sequence_len = self.model.hparams.sequence_length  # the length of total sequence
        # !!! NEED TO BE MODIFIED ACCORDING TO THE IMPLEMENTATION OF YOUR MODEL end

        self.input_idx = 0

        self.batch_size = batch_size

    def main(self, checkpoint_path):
        raw_data = self.go_through_dataset(checkpoint_path)
        q_samples_seq, q_params_seq = raw_data[:2]
        mask = self.build_mask(*raw_data[2:])
        mig, mig_properties, mig_properties_parts, mi = self.compute_mig(q_samples_seq, q_params_seq, mask)
        return mig, mig_properties, mig_properties_parts, mi

    def go_through_dataset(self, checkpoint_path):
        raise NotImplementedError

    def build_mask(self, *args):
        raise NotImplementedError

    def compute_mig(self, q_samples_seq, q_params_seq, mask):
        # compute MIG
        graph_dict = self.log_q_z_batch_sum_graph()

        # select the first predicted frame
        q_samples = np.take(q_samples_seq, self.test_frame, axis=-2).squeeze()
        q_params = np.take(q_params_seq, self.test_frame, axis=-2).squeeze()

        # mutual information
        mi_list = []
        entropie_z = self.compute_entropie(q_samples, q_params, graph_dict)

        for i in range(len(self.num_value_list)):
            print('evaluate %dth proproties' % i)
            cond_entropie_z_v = 0
            p_value_list = []
            for j in range(self.num_value_list[i]):
                select_idx = mask[i][j]
                if sum(select_idx) > 0:
                    p_value_list.append(sum(select_idx) / len(select_idx))
                    q_samples_scale = q_samples[select_idx]
                    q_params_scale = q_params[select_idx]
                    cond_entropie_z_v += \
                        self.compute_entropie(q_samples_scale, q_params_scale, graph_dict) * p_value_list[-1]
            mi = entropie_z - cond_entropie_z_v
            entropie_v = - sum(np.log(p_value_list) * p_value_list)
            mi_norm = mi / entropie_v
            mi_list.append(mi_norm)

        mi_sort_list = [np.sort(mi) for mi in mi_list]

        mig_properties_parts = [[mi_sort[-1], mi_sort[-2]] for i, mi_sort in enumerate(mi_sort_list)]
        mig_properties = [mig_properties_part[0] - mig_properties_part[1] for mig_properties_part in
                          mig_properties_parts]
        mig = np.mean(mig_properties)
        return mig, mig_properties, mig_properties_parts, mi_list

    def load_new_model(self, checkpoint_path):
        """
        !!! NEED TO BE MODIFIED ACCORDING TO THE IMPLEMENTATION OF YOUR MODEL
        This function is used to load a new trained net saved in checkpoint_path
        and reinitialize the dataset for the next evaluation.
        """
        self.model.restore(self.sess, checkpoint_path)
        self.input_idx = 0
        self.sess.run(self.inputs_init)

    def reset_graph(self):
        self.sess.close()
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

    def fetch_inputs(self):
        try:
            input_results = self.sess.run(self.inputs)
        except tf.errors.OutOfRangeError:
            print("OutOfRangeError")
            return None

        feed_dict = {input_ph: input_results[name] for name, input_ph in self.input_phs.items()}
        self.input_idx += self.batch_size
        print("fetch sample %d" % self.input_idx)

        return feed_dict, input_results

    def run_model(self, input_feed_dict):
        """
        !!! NEED TO BE MODIFIED ACCORDING TO THE IMPLEMENTATION OF YOUR MODEL
        This function should output the samples of latent code
        (shape: [batch_size, length of output sequence, #dimention of latent code])
        and the parameters of the latent code distribution
        (shape: [batch_size, length of output sequence, 2 * #dimention of latent code]
        """
        input_feed_dict = input_feed_dict.copy()

        fetch_var = [self.model.outputs['q_samples'],
                     self.model.outputs['q_params']]
        if 'q_params' in self.model.outputs.keys():
            fetch_var.append(self.model.outputs['q_params'])
        fetch_var = self.sess.run(fetch_var, feed_dict=input_feed_dict)
        q_samples, q_params = fetch_var

        return q_samples, q_params

    def log_q_z_batch_sum_graph(self):
        """
        input z_i, q(*|n_j)
        compute q(z_i) = mean_j q(z_i|n_j)
        return sum_i log(q(z_i))
        """
        feed_q_samples_batch = tf.placeholder(tf.float32, [None, self.latent_dim])
        reshape_q_samples_batch = tf.expand_dims(feed_q_samples_batch, axis=1)
        feed_q_params = tf.placeholder(tf.float32, [None, self.latent_dim * 2])
        num_data = tf.cast(tf.shape(feed_q_params)[0], dtype=tf.float32)
        log_num_data = tf.log(num_data)
        log_q_z_cond_ns = umath.norm_log_density(reshape_q_samples_batch, feed_q_params)
        log_q_z_batch = umath.log_sum_exp(log_q_z_cond_ns, axis=1) - log_num_data
        tf_log_q_z_batch_sum = tf.reduce_sum(log_q_z_batch, axis=0)

        graph_dict = {'input': [feed_q_samples_batch, feed_q_params], 'output': tf_log_q_z_batch_sum}
        return graph_dict

    def compute_entropie(self, q_samples, q_params, graph_dict):
        """
        input z_j, q(*|n_j)
        return mean_j -log(q(z_j))
        """
        entropie = 0
        batch_size = 256
        num_data = q_params.shape[0]

        current_idx = 0
        while current_idx < num_data:
            end_idx = current_idx + batch_size if current_idx + batch_size < num_data else num_data
            q_samples_batch = q_samples[current_idx:end_idx]
            log_q_z = self.sess.run(graph_dict['output'],
                               {graph_dict['input'][0]: q_samples_batch, graph_dict['input'][1]: q_params})
            current_idx += batch_size
            entropie += -log_q_z / num_data
        return entropie


def plot_mig(mig_array, beta, save_path=None, name='mig'):
    plt.figure()
    mig_df = pd.DataFrame(mig_array, columns=beta).melt(var_name='beta', value_name='MIG')
    sns_plt = sns.lineplot(x='beta', y='MIG', data=mig_df)
    if save_path is not None:
        fig = sns_plt.get_figure()
        fig.savefig(os.path.join(save_path, name + '.pdf'))


def plot_mig_properties(mig_properties_array, properties_list, beta, save_path=None, name='mig_properties'):
    plt.figure()
    property_mig_list = []
    for property_name, property_mig in zip(properties_list, mig_properties_array):
        property_mig_list.append(
            pd.DataFrame(property_mig, columns=beta).melt(var_name='beta', value_name='MIG'))
        property_mig_list[-1]['property'] = property_name
    property_mig_df = pd.concat(property_mig_list)
    sns_plt = sns.lineplot(x='beta', y='MIG', hue='property', data=property_mig_df)
    if save_path is not None:
        fig = sns_plt.get_figure()
        fig.savefig(os.path.join(save_path, name + '.pdf'))
