import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--runner", action="store_true",
                    default=False, help="start rl runner with dl")
parser.add_argument("--runner-rp", action="store_true", default=False,
                    help="start rl runner with replay buffer with dl")
parser.add_argument("--runner-master", action="store_true", default=False,
                    help="start rl runner with the baseline of the masters project")
parser.add_argument("--real-failures", action="store_true", default=False, help="Restrict the agent to only give actions on failed components.")
parser.add_argument("--forbid-duplicate-action", action="store_true", default=False, help="Forbid the agent to return the same action twice for a component.")
parser.add_argument("--shared-replay-buffer", action="store_true", help="Share one replay buffer between all agents")
parser.add_argument("--disable-random-batch", action="store_true", help="Disbale random sampling from replay buffer.")
parser.add_argument("--batch-size", type=int, default=20, help="Sample size from replay buffer.")
parser.add_argument("--balanced-sampling", action="store_true", help="Balance positive and negative signals when sampling from replay buffer.")
parser.add_argument("--positive-sampling", action="store_true", help="Only give positive signals when sampling from replay buffer.")
parser.add_argument("--wandb", action="store_true", help="Log with wandb")
parser.add_argument("--on-policy", action="store_true", help="Enable on policy training.")
parser.add_argument("--on-off-policy", action="store_true", help="Enable on-off policy training.")
parser.add_argument("--use-exploration", action="store_true", default=False, help="Use exploration with guard for the master project agent" )
parser.add_argument("--off-policy-factor", type=float, default=0.1, help="Factor of off-policy in loss calculation.")
parser.add_argument("--transfer-strategy", type=str, default="replace", help="Transfer strategy to use for the agents after half the episodes are over. e.g. replace, combine, etc,")
parser.add_argument("--n-retrain", type=int, default=10, help="Number of iterations to retrain transfer learning target before continuing normal flow.")
parser.add_argument("--retrain-cat", action="store_true", help="Use categorical loss to retrain transfer learning target instead of reward.")
parser.add_argument("--off-policy-loss-cat", action="store_true", help="Use categorical cross entropy loss as off-policy loss function in on-off-policy loss computation.")
parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
args = parser.parse_args()
