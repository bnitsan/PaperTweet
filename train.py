import wandb
from rich.table import Column, Table
from rich import box
from rich.console import Console

import sys  # insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'models')

from cmd_utils import parse_args
from utils import read_yml
from model import CondGenModel
import pytorch_lightning as light
from utils import initialize_wandb
from pytorch_lightning.loggers import WandbLogger
import os

# define a rich console logger
console = Console(record=True)
model_params = read_yml('config.yml')['config']

def main(argv):
 
    console.log(f"""[Model]: {model_params["MODEL"]}...\n""")

    # parse parameters of command line execution
    input_file, output_file, model_path, run_name = parse_args(argv)
    model_params.update({"DATA_PATH": input_file})

    model = CondGenModel(model_params, console)

    # console.log(f"""[Task]: Going to train \n""")
    # train_params = model_params
    train_params = dict(
        max_epochs=model_params["TRAIN_EPOCHS"],
        # early_stop_callback=False,
        gradient_clip_val=1.0,
        log_every_n_steps=5,
    )

    wandb_logger = None
    if model_params["WANDB"]:
        # wandb = initialize_wandb(model_params)
        # wandb.config.update()
        if run_name == '':
            run_name = None
        wandb_logger = WandbLogger(project="papertweet", name=run_name)

    trainer = light.trainer.trainer.Trainer(accelerator='gpu', devices=1,
                                            # dir=model_params["OUTPUT_DIR_MODELS"],
                                            accumulate_grad_batches=model_params["TRAIN_BATCH_ACCUM"],
                                            # detect_anomaly=True,  # track_grad_norm=2,
                                            logger=wandb_logger,  **train_params)  #
    trainer.fit(model)

    trainer.save_checkpoint(os.path.join(model_params["OUTPUT_DIR_MODELS"], model_path+".ckpt"), weights_only=True)

if __name__ == "__main__":
    main(sys.argv[1:])

