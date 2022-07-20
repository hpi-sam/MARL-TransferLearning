import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--runner", action="store_true",
                    default=False, help="start rl runner with dl")
parser.add_argument("--runner-rp", action="store_true", default=False,
                    help="start rl runner with replay buffer with dl")
parser.add_argument("--runner-master", action="store_true", default=False,
                    help="start rl runner with the baseline of the masters project")
parser.add_argument("--real_failures", action="store_true", default=False, help="Restrict the agent to only give actions on failed components.")
parser.add_argument("--forbid_duplicate_action", action="store_true", default=False, help="Forbid the agent to return the same action twice for a component.")
parser.add_argument("--shared_replay_buffer", action="store_true", help="Share one replay buffer between all agents")
parser.add_argument("--wandb", action="store_true", help="Log with wandb")
args = parser.parse_args()