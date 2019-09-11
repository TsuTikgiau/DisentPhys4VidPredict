# This file is derived from Alex X. Lee's SAVP project
# (https://github.com/alexlee-gk/video_prediction) with the MIT license.
# The license can be found in LICENSE.md file under the root.


import argparse
import errno
import json
import os

import numpy as np
import tensorflow as tf

from video_prediction import datasets, models  # !!! NEED TO BE MODIFIED ACCORDING TO THE IMPLEMENTATION OF YOUR MODEL
from video_prediction.eval_mig import plot_mig, plot_mig_properties, get_mig_eval


def initial():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="either a directory containing subdirectories "
                                                                     "train, val, test, etc, or a directory containing "
                                                                     "the tfrecords")

    parser.add_argument("--checkpoint_root",
                        help="directory where all the checkpoints are saved", type=str, required=True)
    parser.add_argument("--prefix",
                        help="prefix of each checkpoint", type=str, default="tc")

    parser.add_argument("--dataset", type=str, help="dataset class name [slide, wall, collision]")
    parser.add_argument("--model", type=str, default="savp", help="model class name")
    parser.add_argument("--model_hparams", type=str,
                        help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size. note that in the current version, "
                             "dataset size should be divisible by batch size")
    parser.add_argument("--beta_list", type=int, nargs="+",
                        help="list of beta")
    parser.add_argument("--num_exp", type=int, required=True,
                        help="number of experiments per beta")
    parser.add_argument("--result_dir",
                        help="directory where to save the evaluated mig")
    parser.add_argument("--plot_result", action='store_ture',
                        help="whether to plot mig after evaluation")

    args = parser.parse_args()

    model_hparams_dict = {}

    checkpoint_dir = os.path.join(os.path.normpath(args.checkpoint_root), args.prefix + '%s_0' % args.beta_list[0])
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
    with open(os.path.join(checkpoint_dir, "options.json")) as f:
        print("loading options from checkpoint %s" % checkpoint_dir)
        options = json.loads(f.read())
        args.dataset = args.dataset or options['dataset']
        args.model = args.model or options['model']

    VideoDataset = datasets.get_dataset_class(args.dataset)
    dataset = VideoDataset(args.input_dir, mode='val', num_epochs=1)

    # !!! NEED TO BE MODIFIED ACCORDING TO THE IMPLEMENTATION OF YOUR MODEL
    # Model Initialization
    try:
        with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
            model_hparams_dict = json.loads(f.read())
            model_hparams_dict.pop('num_gpus', None)  # backwards-compatibility
    except FileNotFoundError:
        print("model_hparams.json was not loaded because it does not exist")

    def override_hparams_dict(dataset):
        hparams_dict = dict(model_hparams_dict)
        hparams_dict['context_frames'] = dataset.hparams.context_frames
        hparams_dict['sequence_length'] = dataset.hparams.sequence_length
        hparams_dict['repeat'] = dataset.hparams.time_shift
        return hparams_dict

    VideoPredictionModel = models.get_model_class(args.model)
    model = VideoPredictionModel(mode='test', hparams_dict=override_hparams_dict(dataset),
                                 hparams=args.model_hparams)
    # !!! NEED TO BE MODIFIED ACCORDING TO THE IMPLEMENTATION OF YOUR MODEL end

    return model, dataset, args


# ======================================================================
#                               START
# ======================================================================

def mig_evaluation(model, dataset, args):
    """
    Assume we have trained several networks per beta value (beta is the penalty on the
    KL in beta-VAE or on the total correlation in beta-TCVAE) and also we have different
    beta values. The filename of saved network parameters is in the form
    "prefix"+"beta_value"+"_"+"num_exp". For example, 'blabla2_0'. This function will
    load all the trained networks sequentially and compute

    mig_array: their average mutual information gap (size: [#beta, #exp]),
    mig_properties_array: their mutual information gap per property (size: [#properties, #beta, #exp]),
    mig_properties_parts_array: their largest and second largest mutual information per property
    (size: [2, #properties, #beta, #exp]),
    mi_array: their all mutual information per property (size: [#latent, #properties, #beta, #exp]).

    """
    eval_mig = get_mig_eval(args.dataset)(model, dataset, args.batch_size)

    mig_list = []
    mig_properties_list = []
    mig_properties_parts_list = []
    mi_list = []
    for i in args.beta_list:
        prefix = args.prefix + str(i)
        sub_mig_list = []
        sub_mig_properties_list = []
        sub_mig_properties_parts_list = []
        sub_mi_list = []
        for j in ['_' + str(i) for i in range(args.num_exp)]:
            checkpoint_path = os.path.join(args.checkpoint_root, prefix + j)
            mig, mig_properties, mig_properties_parts, mi = eval_mig.main(checkpoint_path)
            sub_mig_list.append(mig)
            sub_mig_properties_list.append(mig_properties)
            sub_mig_properties_parts_list.append(mig_properties_parts)
            sub_mi_list.append(mi)
        mig_list.append(sub_mig_list)
        mig_properties_list.append(sub_mig_properties_list)
        mig_properties_parts_list.append(sub_mig_properties_parts_list)
        mi_list.append(sub_mi_list)

        # convert and save mig checkpoint
        mig_array = np.array(mig_list).transpose()
        mig_properties_array = np.array(mig_properties_list).transpose(2, 1, 0)
        mig_properties_parts_array = np.array(mig_properties_parts_list).transpose(3, 2, 1, 0)
        mi_array = np.array(mi_list).transpose(3, 2, 1, 0)

        np.save(os.path.join(args.result_dir, 'mig.npy'), mig_array)
        np.save(os.path.join(args.result_dir, 'mig_properties.npy'), mig_properties_array)
        np.save(os.path.join(args.result_dir, 'mig_properties_parts.npy'), mig_properties_parts_array)
        np.save(os.path.join(args.result_dir, 'mi.npy'), mi_array)

    if args.plot_result:
        # plot
        plot_mig(mig_array, args.beta_list, args.result_dir)
        plot_mig_properties(mig_properties_array, eval_mig.properties_list, args.beta_list, args.result_dir)
        plot_mig_properties(mig_properties_parts_array[0], eval_mig.properties_list,
                            args.beta_list, args.result_dir, name='mi_properties_1')
        plot_mig_properties(mig_properties_parts_array[1], eval_mig.properties_list,
                            args.beta_list, args.result_dir, name='mi_properties_2')

    return mig_array, mig_properties_array, mig_properties_parts_array, mi_array


def main():
    model, dataset, args = initial()
    mig_evaluation(model, dataset, args)


if __name__ == '__main__':
    main()
