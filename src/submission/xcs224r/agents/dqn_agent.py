import numpy as np
import pdb
from typing import Any, Dict, List, Tuple, Union

from xcs224r.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from xcs224r.policies.argmax_policy import ArgMaxPolicy
from xcs224r.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env: Any, agent_params: Dict[str, Any]) -> None:

        self.env = env
        self.agent_params = agent_params
        self.batch_size: int = agent_params['batch_size']
        # import ipdb; ipdb.set_trace()
        self.last_obs, _ = self.env.reset()

        self.num_actions: int = agent_params['ac_dim']
        self.learning_starts: int = agent_params['learning_starts']
        self.learning_freq: int = agent_params['learning_freq']
        self.target_update_freq: int = agent_params['target_update_freq']

        self.replay_buffer_idx: Any = None
        self.exploration: PiecewiseSchedule = agent_params['exploration_schedule']
        self.optimizer_spec: Any = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(self.critic)

        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'])
        self.t: int = 0
        self.num_param_updates: int = 0

    def add_to_replay_buffer(self, paths: List[Dict[str, Any]]) -> None:
        pass

    def step_env(self) -> None:
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """
        raise NotImplementedError
        # Not needed for this homework

    ####################################
    ####################################

    def sample(self, batch_size: int) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[list, list, list, list, list]]:
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no: np.ndarray, ac_na: np.ndarray, re_n: np.ndarray, next_ob_no: np.ndarray, terminal_n: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError
        # Not needed for this homework

    ####################################
    ####################################