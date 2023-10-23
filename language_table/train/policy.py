 # coding=utf-8
# Copyright 2023 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyPolicy for BC Jax."""

from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.specs import tensor_spec
import tensorflow as tf

EPS = jnp.finfo(jnp.float32).eps


class BCJaxPyPolicy(py_policy.PyPolicy):
  """Runs inference with a BC policy."""

  def __init__(self, time_step_spec, action_spec, model,network_state,
               rng, params=None, action_statistics=None):
    super(BCJaxPyPolicy, self).__init__(time_step_spec, action_spec)
    self.model = model
    self.network_state = network_state
    self.rng = rng
    #
    # if params is not None and action_statistics is not None:
    #   variables = {
    #       "params": params,
    #       "batch_stats": {}
    #   }
    # else:
    #   state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)
    #   variables = {
    #       "params": state_dict["params"],
    #       "batch_stats": state_dict["batch_stats"]
    #   }

    # if action_statistics is not None:
    #   self.action_mean = np.array(action_statistics["mean"])
    #   self.action_std = np.array(action_statistics["std"])
    # else:
    #   # We can load the observation and action statistics from the state dict.
    #   self.action_mean = np.array(
    #       state_dict["norm_info"]["action_statistics"]["mean"])
    #   self.action_std = np.array(
    #       state_dict["norm_info"]["action_statistics"]["std"])
    #
    #   self._rgb_mean = jnp.array(
    #       state_dict["norm_info"]["observation_statistics"]["rgb"]["mean"])
    #   self._rgb_std = jnp.array(
    #       state_dict["norm_info"]["observation_statistics"]["rgb"]["std"])
    #
    # self.variables = variables

    # self._run_action_inference_jit = jax.jit(self._run_action_inference)
  def _run_action_inference(self, observation):
    
    # Add a batch dim.
    observation = jax.tree_map(lambda x: jnp.expand_dims(x, 0), observation)
    print(observation)
    # 构造模型所需的输入
    # print(observation["rgb"].shape)
    # exit()
    # resized_tensor = tf.image.resize(observation["rgb"] size=(256, 456))
    observation_input = {}
    observation_input['image'] = tf.convert_to_tensor(observation['rgb_sequence'])
    observation_input['natural_language_embedding'] = tf.convert_to_tensor(observation['instruction_tokenized_use'])
    observation_input['natural_language_embedding'] = tf.squeeze(observation_input['natural_language_embedding'], axis=2)

    output,_ = self.model(observation_input,step_type=None, network_state=self.network_state, training=False)
    # print(output['effector_target_translation'].numpy()[0])
    # print(observation['effector_translation'][:,:,:])
    action = output['effector_target_translation'].numpy()[0] - observation['effector_translation'][0,5,:]
    action = action * 0.1
    action = np.clip(action,self.action_spec.minimum,self.action_spec.maximum)  #zy
    return action

  def _action(self, time_step, policy_state=(), seed=0):
    observation = time_step.observation
    action = self._run_action_inference(observation)
    return policy_step.PolicyStep(action=action)