import os
import torch
import numpy as np
import pandas as pd
import sys  # insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'models')

from cmd_utils import parse_args
from utils import read_yml, get_test_dataloaders
from model import CondGenModel

from rich.console import Console

# define a rich console logger
console = Console(record=True)

model_params = read_yml('config.yml')['config']


def infer(model, inference_params, input_file, keyword_column):
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    
    model.eval()

    inference_loader = get_test_dataloaders(input_file, model.tokenizer, model_params, **inference_params)

    predictions = []
    with torch.no_grad():
        for idx, batch in enumerate(inference_loader):
            output = model.test_step(batch, 0)  # 0 for batch_idx which is a placeholder for now
            preds = output["preds"]
            predictions.extend(preds)
            if idx % 10 == 0:
                console.print(f'Completed {idx}')
    test_df = pd.read_csv(input_file)
    test_df[keyword_column] = predictions
    return test_df


def main(argv):
    keyword_column_baseline = "BASELINE"
    keyword_column_gen = "Generated_Tweets"

    console.log(f"""[Model]: {model_params["MODEL"]}...\n""")

    # parse parameters of command line execution
    input_file, output_file, model_path, run_name = parse_args(argv)
    model_params.update({"DATA_PATH": input_file})

    inference_params = {
        "batch_size": model_params["TEST_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": model_params["NUM_WORKERS"],
    }

    console.log(f"""[Inference]: On fine-tuned model...\n""")

    model = CondGenModel.load_from_checkpoint(os.path.join(model_params["OUTPUT_DIR_MODELS"], model_path+".ckpt"), model_params=model_params, console=console)
    test_df = infer(model, inference_params, input_file, keyword_column_gen)

    if model_params["BASELINE_FLAG"]:
        model_params.update({"PREFIX_TO_PROMPT": model_params["BASELINE_PREFIX_TO_PROMPT"]})
        console.log(f"""[Inference]: On baseline model...\n""")

        model = CondGenModel(model_params, console)
        baseline_df = infer(model, inference_params, input_file, keyword_column_baseline)
        test_df[keyword_column_baseline] = baseline_df[keyword_column_baseline]

    print(os.path.join(model_params["OUTPUT_DIR"], output_file))
    test_df.to_csv(os.path.join(model_params["OUTPUT_DIR"], output_file))


if __name__ == "__main__":
    main(sys.argv[1:])
