import numpy as np
from video_prediction.eval_mig.basis_mig import EvalMigBasis


class EvalMigCollision(EvalMigBasis):

    def __init__(self, model, dataset, batch_size):
        super(EvalMigCollision, self).__init__(model, dataset, batch_size)

        self.setting_len = self.inputs['setting_left'].shape[1].value

        self.properties_list = ['shape_left', 'scale_left', 'material_left', 'speed_left', 'position_left',
                                'shape_right', 'scale_right', 'material_right', 'speed_right', 'position_right']
        self.num_value_list = [5, 7, 5, 32, 7, 5, 7, 5, 32, 7]  # shape, scale, material, speed, position
        self.num_data = 2500
        self.test_frame = [3]

    def go_through_dataset(self, checkpoint_path):
        self.load_new_model(checkpoint_path)
        q_samples_seq = np.zeros([self.num_data, self.sequence_len - 1, self.latent_dim])
        q_params_seq = np.zeros([self.num_data, self.sequence_len - 1, self.latent_dim * 2])
        setting_left = np.zeros([self.num_data, self.setting_len])
        setting_right = np.zeros([self.num_data, self.setting_len])
        speed_left = np.zeros([self.num_data, len(self.test_frame)])
        speed_right = np.zeros([self.num_data, len(self.test_frame)])
        position_left = np.zeros([self.num_data, len(self.test_frame)])
        position_right = np.zeros([self.num_data, len(self.test_frame)])

        while self.input_idx < self.num_data:
            input_feed_dict, input_dict = self.fetch_inputs()
            q_samples_seq[self.input_idx - self.batch_size:self.input_idx, :, :], \
            q_params_seq[self.input_idx - self.batch_size:self.input_idx, :, :] \
                = self.run_model(input_feed_dict)
            setting_left[self.input_idx - self.batch_size:self.input_idx, :] = input_dict['setting_left']
            setting_right[self.input_idx - self.batch_size:self.input_idx, :] = input_dict['setting_right']
            speed_left[self.input_idx - self.batch_size:self.input_idx, :] = input_dict['speed_left_x'][:,
                                                                             self.test_frame, 0]
            speed_right[self.input_idx - self.batch_size:self.input_idx, :] = input_dict['speed_right_x'][:,
                                                                              self.test_frame, 0]

            position_left[self.input_idx - self.batch_size:self.input_idx, :] = input_dict['position_left_x'][:,
                                                                                self.test_frame, 0]
            position_right[self.input_idx - self.batch_size:self.input_idx, :] = input_dict['position_right_x'][:,
                                                                                 self.test_frame, 0]

        assert self.input_idx == self.num_data

        return q_samples_seq, q_params_seq, setting_left, setting_right, speed_left, speed_right, position_left, position_right

    def build_mask(self, setting_left, setting_right, speed_left, speed_right, position_left, position_right):
        # build bool mask
        mask = []

        # -----------left------------
        # shape
        sub_mask = []
        for i in range(self.num_value_list[0]):
            sub_mask.append(setting_left[:, 0] == i)
        mask.append(np.stack(sub_mask, 0))
        # scale
        sub_mask = []
        for i in range(self.num_value_list[1]):
            sub_mask.append(setting_left[:, 1] == i)
        mask.append(np.stack(sub_mask, 0))
        # material
        sub_mask = []
        for i in range(self.num_value_list[2]):
            sub_mask.append(setting_left[:, 2] == i)
        mask.append(np.stack(sub_mask, 0))
        # speed
        sub_mask = []
        for i in np.linspace(-15.5, 15.5, self.num_value_list[3]):
            sub_mask.append(((speed_left - i < 0.50001) * (speed_left - i >= -0.5))[:, 0])
        mask.append(np.stack(sub_mask, 0))
        # position
        sub_mask = []
        for i in np.linspace(0.5, 6.5, self.num_value_list[4]):
            sub_mask.append(((position_left - i < 0.50001) * (position_left - i >= -0.5))[:, 0])
        mask.append(np.stack(sub_mask, 0))

        # -----------right-----------
        # shape
        sub_mask = []
        for i in range(self.num_value_list[5]):
            sub_mask.append(setting_right[:, 0] == i)
        mask.append(np.stack(sub_mask, 0))
        # scale
        sub_mask = []
        for i in range(self.num_value_list[6]):
            sub_mask.append(setting_right[:, 1] == i)
        mask.append(np.stack(sub_mask, 0))
        # material
        sub_mask = []
        for i in range(self.num_value_list[7]):
            sub_mask.append(setting_right[:, 2] == i)
        mask.append(np.stack(sub_mask, 0))
        # speed
        sub_mask = []
        for i in np.linspace(-15.5, 15.5, self.num_value_list[8]):
            sub_mask.append(((speed_right - i < 0.50001) * (speed_right - i >= -0.5))[:, 0])
        mask.append(np.stack(sub_mask, 0))
        # position
        sub_mask = []
        for i in np.linspace(0.5, 6.5, self.num_value_list[9]):
            sub_mask.append(((position_right - i < 0.50001) * (position_right - i >= -0.5))[:, 0])
        mask.append(np.stack(sub_mask, 0))

        return mask
