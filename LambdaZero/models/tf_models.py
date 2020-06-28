import time
from ray.rllib.utils import try_import_tf
import numpy as np
tf = try_import_tf()

from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.tf.misc import normc_initializer

# from LambdaZero.utils import RunningMeanStd

class MolActorCritic_tfv1(DistributionalQTFModel, TFModelV2):
    def __init__(self,obs_space, action_space, num_outputs, model_config, name, **kw):
        super(MolActorCritic_tfv1, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)

        ospace = obs_space.original_space
        mol_fp_len = ospace["mol_fp"].shape[0]
        stem_fp_len = ospace["stem_fps"].shape[1]
        max_stems = ospace["stem_fps"].shape[0]
        max_jbonds = ospace["jbond_fps"].shape[0]
        n_blocks = (ospace["action_mask"].shape[0] - 1 - ospace["jbond_fps"].shape[0]) // ospace["stem_fps"].shape[0]
        self.max_steps = ospace["num_steps"].n

        # shared layers
        mol_embed = tf.keras.layers.Input(shape=(mol_fp_len,))
        mol_embed_2 = tf.keras.layers.Dense(stem_fp_len,
                                          kernel_initializer=normc_initializer(1.0),
                                          activation=tf.nn.tanh)(mol_embed)

        # actor layers
        stop_logit = tf.keras.layers.Dense(1,
                                           activation=None,
                                           kernel_initializer=normc_initializer(0.01))(mol_embed_2)#[:,0]
        #
        jbond_embed = tf.keras.layers.Input(shape=(max_jbonds, stem_fp_len,))
        mol_embed_2_ = tf.tile(tf.expand_dims(mol_embed_2, 1), [1, max_jbonds, 1])
        jbond_embed_2 = tf.keras.layers.Concatenate(axis=2)([jbond_embed, mol_embed_2_, jbond_embed * mol_embed_2_])
        break_logits = tf.keras.layers.Dense(1,
                                           activation=None,
                                           kernel_initializer=normc_initializer(0.01))(jbond_embed_2)
        break_logits = tf.keras.layers.Flatten()(break_logits)
        #
        stem_embed = tf.keras.layers.Input(shape=(max_stems, stem_fp_len,))
        mol_embed_2_ = tf.tile(tf.expand_dims(mol_embed_2, 1), [1, max_stems, 1])
        stem_embed_2 = tf.keras.layers.Concatenate(axis=2)([stem_embed, mol_embed_2_, stem_embed * mol_embed_2_])
        add_logits = tf.keras.layers.Dense(n_blocks,
                                           activation=None,
                                           kernel_initializer=normc_initializer(0.01))(stem_embed_2)
        add_logits = tf.keras.layers.Flatten()(add_logits)

        # critic layers
        value_logit = tf.keras.layers.Dense(1,
                                           activation=None,
                                           kernel_initializer=normc_initializer(0.01))(mol_embed_2)[:,0]
        action_logits = tf.keras.layers.Concatenate(axis=1)([stop_logit, break_logits, add_logits])

        # build keras model
        self.actor_critic = tf.keras.Model([mol_embed, jbond_embed, stem_embed], [action_logits, value_logit])
        self.register_variables(self.actor_critic.variables)

        # RND
        if 'rnd_weight' in model_config['custom_options'].keys():
            self.rnd_weight = model_config['custom_options']['rnd_weight']
        else:
            self.rnd_weight = 0
        # import pdb;pdb.set_trace();
        if self.rnd_weight > 0:
            # Target
            rnd_target_inp = tf.keras.layers.Input(shape=(7619,))
            rnd_target_1 = tf.keras.layers.Dense(256,
                                           activation=tf.nn.relu,
                                           kernel_initializer=normc_initializer(0.01))(rnd_target_inp)
            rnd_target_out = tf.keras.layers.Dense(256,
                                           activation=None,
                                           kernel_initializer=normc_initializer(0.01))(rnd_target_1)
            rnd_target_1.trainable=False
            rnd_target_out.trainable=False
            self.rnd_target = tf.keras.Model([rnd_target_inp], [rnd_target_out])
            # Predictor
            rnd_predictor_inp = tf.keras.layers.Input(shape=(7619,))
            rnd_predictor_1 = tf.keras.layers.Dense(256,
                                           activation=tf.nn.relu,
                                           kernel_initializer=normc_initializer(0.01))(rnd_predictor_inp)
            rnd_predictor_out = tf.keras.layers.Dense(256,
                                           activation=None,
                                           kernel_initializer=normc_initializer(0.01))(rnd_predictor_1)
            self.rnd_predictor = tf.keras.Model([rnd_predictor_inp], [rnd_predictor_out])
            self.flatten_obs = tf.keras.layers.Flatten()
            # register only predictor weights to be updated by optim.
            self.register_variables(self.rnd_predictor.variables)
            self.register_variables(self.rnd_target.variables)
    def forward(self, input_dict, state, seq_lens):

        # Extract the available actions tensor from the observation.
        mol_fp = input_dict["obs"]["mol_fp"]
        jbond_fps = input_dict["obs"]["jbond_fps"]
        stem_fps = input_dict["obs"]["stem_fps"]
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, self._value = self.actor_critic(inputs=[mol_fp, jbond_fps, stem_fps])

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state

    
    @override(TFModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        if self.rnd_weight > 0:
            obs = self.flatten_obs(loss_inputs['obs'])
            # obs = tf.clip_by_value((obs_r - self.rnd_obs_stats.mean) / (self.rnd_obs_stats.std), -5, 5)
            target_reward = self.rnd_target(obs)
            predictor_reward = self.rnd_predictor(obs)
            self.rnd_loss = tf.keras.losses.mean_squared_error(target_reward, predictor_reward)
            # self.rnd_loss = (rnd_loss - self.rnd_rew_stats.mean) / (self.rnd_rew_stats.std)
            self.policy_loss = policy_loss
            # self.rnd_obs_stats.update(obs_r)
            # self.rnd_rew_stats.update(self.rnd_loss)
            return policy_loss + self.rnd_weight * self.rnd_loss
        else:
            return policy_loss

    def value_function(self):
        return self._value

    def custom_stats(self):
        return {
            "rnd_loss": self.rnd_loss,
            "policy_loss": self.policy_loss
        }