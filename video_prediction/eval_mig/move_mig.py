import numpy as np
from video_prediction.eval_mig.basis_mig import EvalMigBasis


class EvalMigMove(EvalMigBasis):

    def __init__(self, model, dataset, batch_size):
        super(EvalMigMove, self).__init__(model, dataset, batch_size)

        self.setting_len = self.inputs['setting'].shape[1].value

        self.properties_list = ['shape', 'scale', 'friction', 'speed', 'position']
        self.num_value_list = [5, 10, 8, 10, 9]  # shape, scale, friction, speed, position
        self.num_data = 2944
        self.test_frame = [3]

    def go_through_dataset(self, checkpoint_path):
        self.load_new_model(checkpoint_path)
        q_samples_seq = np.zeros([self.num_data, self.sequence_len - 1, self.latent_dim])
        q_params_seq = np.zeros([self.num_data, self.sequence_len - 1, self.latent_dim * 2])
        setting = np.zeros([self.num_data, self.setting_len])
        speed = np.zeros([self.num_data, len(self.test_frame)])
        position = np.zeros([self.num_data, len(self.test_frame)])

        while self.input_idx < self.num_data:
            input_feed_dict, input_dict = self.fetch_inputs()
            q_samples_seq[self.input_idx - self.batch_size:self.input_idx, :, :], \
            q_params_seq[self.input_idx - self.batch_size:self.input_idx, :, :] \
                = self.run_model(input_feed_dict)
            setting[self.input_idx - self.batch_size:self.input_idx, :] = input_dict['setting']
            speed[self.input_idx - self.batch_size:self.input_idx, :] = input_dict['speed'][:, self.test_frame, 0]
            position[self.input_idx - self.batch_size:self.input_idx, :] = input_dict['position'][:, self.test_frame, 0]

        assert self.input_idx == self.num_data

        return q_samples_seq, q_params_seq, setting, speed, position

    def build_mask(self, setting, speed, position):
        # build bool mask
        mask = []
        # shape
        sub_mask = []
        for i in range(self.num_value_list[0]):
            sub_mask.append(setting[:, 0] == i)
        mask.append(np.stack(sub_mask, 0))
        # scale
        sub_mask = []
        for i in range(self.num_value_list[1]):
            sub_mask.append(setting[:, 1] == i)
        mask.append(np.stack(sub_mask, 0))
        # friction
        sub_mask = []
        for i in range(self.num_value_list[2]):
            sub_mask.append(setting[:, 2] == i)
        mask.append(np.stack(sub_mask, 0))
        # speed
        sub_mask = []
        for i in np.linspace(0.5, 9.5, self.num_value_list[3]):
            sub_mask.append(((speed - i < 0.500001) * (speed - i >= -0.5))[:, 0])
        mask.append(np.stack(sub_mask, 0))
        # position
        sub_mask = []
        for i in np.linspace(0.5, 8.5, self.num_value_list[4]):
            sub_mask.append(((position - i < 0.500001) * (position - i >= -0.5))[:, 0])
        mask.append(np.stack(sub_mask, 0))
        return mask

