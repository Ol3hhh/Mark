import gymnasium as gym

from . import agents

gym.register(
    id="Mark-Flat-v2-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mark_v2_env_cfg:MarkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MarkPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_cfg:MarkPPORunnerCfg",
    },
)

gym.register(
    id="Mark-Flat-v2-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mark_v2_env_cfg:MarkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MarkPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_cfg:MarkPPORunnerCfg",
    },
)

# For enviroment tests RA- random agent
gym.register(
    id="Mark-Flat-v2-RA",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mark_v2_env_cfg:MarkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MarkPPORunnerCfg",
    },
)

gym.register(
    id="Mark-Flat-v2-Check",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mark_v2_env_cfg:MarkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MarkPPORunnerCfg",
    },
)
