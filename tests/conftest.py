import os

# Tests must never create W&B runs.
os.environ["DISPATCH_WANDB"] = "0"
