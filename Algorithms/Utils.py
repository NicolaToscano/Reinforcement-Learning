# Libraries
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from gym_unity.envs import UnityEnv

class Utils():

    def getActionStateSize(env):
        brain_name = env._env.get_agent_groups()[0]
        a = env._env.get_agent_group_spec(brain_name)
        action_size = a.discrete_action_branches[0]
        state_size = a.observation_shapes[0][0]
        print("Action size: ", action_size, "\nState size: ", state_size)
        return action_size, state_size