from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ..mdp.symmetry import compute_symmetric_states as mark_symmetry


@configclass
class MarkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 25
    experiment_name = "mark_v1_flat"

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mark_symmetry,
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        for model in (self.actor, self.critic):
            for attr in (
                "stochastic",
                "init_noise_std",
                "noise_std_type",
                "state_dependent_std",
            ):
                if hasattr(model, attr):
                    delattr(model, attr)
