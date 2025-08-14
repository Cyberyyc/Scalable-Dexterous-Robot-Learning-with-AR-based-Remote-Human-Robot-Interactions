"""Two-stage learning with an application-provided Gymnasium environment."""
import argparse
import importlib
from core.contrastive_sac import CustomSAC
from core.behavior_cloning import load_transitions, pretrain_actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-factory", required=True, help="Import path module:function")
    parser.add_argument("--expert-data", required=True, help="NPZ containing observations and actions")
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--bc-epochs", type=int, default=200)
    parser.add_argument("--save", default="checkpoints/policy")
    args = parser.parse_args()
    module, factory = args.env_factory.split(":")
    env = getattr(importlib.import_module(module), factory)()
    model = CustomSAC("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]},
                      learning_rate=7.3e-4, gamma=0.98, batch_size=512, ent_coef=1.0)
    observations, actions = load_transitions(args.expert_data)
    pretrain_actor(model, observations, actions, epochs=args.bc_epochs)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    env.close()


if __name__ == "__main__":
    main()
