import gymnasium as gym

from . import agents

gym.register(
    id="Mark-Flat-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mark_env_cfg:MarkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MarkPPORunnerCfg",
                # "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_cfg:MarkPPORunnerCfg",
    },
)

gym.register(
    id="Mark-Flat-v1-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mark_env_cfg:MarkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MarkPPORunnerCfg",
                # "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_cfg:MarkPPORunnerCfg",
    },
)
