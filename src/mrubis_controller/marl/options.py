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
args = parser.parse_args()