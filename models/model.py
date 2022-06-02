from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer

import pytorch_lightning as light
from utils import get_train_dataloaders, get_val_dataloaders, train_val_test_path, jaccard
# from transformers import AdamW, Adafactor
# from transformers import AdafactorSchedule
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from utils import get_num_train_steps
from utils import rouge_score
import torch
import numpy as np
import pandas as pd
from pathlib import Path

model_class_dict = {"T5": T5ForConditionalGeneration,
                    "BART": BartForConditionalGeneration,
                    "PEGASUS": PegasusForConditionalGeneration}
tokenizer_class_dict = {"T5": T5Tokenizer,
                        "BART": BartTokenizer,
                        "PEGASUS": PegasusTokenizer}

class CondGenModel(light.LightningModule):
    """
    An implementation of the model module of PyTorch Lightning, superseding various methods;
    """
    def __init__(self, model_params, console):
        super().__init__()

        # random seed
        torch.manual_seed(model_params["SEED"])  # pytorch random seed
        np.random.seed(model_params["SEED"])  # numpy random seed

        self.model_params = model_params
        self.console = console

        # we load a model and tokenizer based on the dictionaries above and model_params["MODEL_CLASS"]
        self.model = model_class_dict[model_params["MODEL_CLASS"]].from_pretrained(model_params["MODEL"])
        self.tokenizer = tokenizer_class_dict[model_params["MODEL_CLASS"]].from_pretrained(model_params["MODEL"], model_max_length=model_params["MAX_SOURCE_TEXT_LENGTH"])

        # add extra special tokens and resize model vocabulary
        special_tokens_dict = model_params["SPECIAL_TOKENS_DICT"]
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # get paths for data
        self.train_path, self.val_path, self.test_path = train_val_test_path(self.model_params["DATA_PATH"])
        if Path(self.train_path).exists():
            self.train_len = len(pd.read_csv(self.train_path))
        else:
            self.console.log(f"""[Training]: False...\n""")
            self.train_len = 0

        # keep a copy of console
        self.console = console
        self.console.log(f"""[Initialization]: Complete...\n""")

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def train_dataloader(self):
        train_dl, train_len = get_train_dataloaders(self.train_path, self.tokenizer, self.model_params)
        self.train_len = train_len
        # self.log(batch_size=self.model_params["TRAIN_BATCH_SIZE"])
        return train_dl

    def val_dataloader(self):
        val_dl = get_val_dataloaders(self.val_path, self.tokenizer, self.model_params)
        return val_dl

    def training_step(self, batch, batch_nb):
        y = batch['target_ids']
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == 0] = -100
        output = self.model(input_ids=batch['source_ids'], attention_mask=batch['source_mask'], labels=lm_labels)
        tensorboard_logs = {'train_loss': output[0]}

        self.log('train_loss', output[0].item())
        return {'loss': output[0], 'log': tensorboard_logs}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.model_params["WEIGHT_DECAY"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        """
        We attempted several options for the optimizer, eventually settling on an untweaked Adam optimizer. 
        """
        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
        #                               lr=self.model_params["LEARNING_RATE"])
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.model_params["LEARNING_RATE"])
        # optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=self.model_params["LEARNING_RATE"])
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=self.model_params["LEARNING_RATE"]
        )

        """
        We attempted a few options for a scheduler, currently using a linear schedule with warmup 
        """
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.model_params["WARMUP_STEPS"],  # 0
            num_training_steps=get_num_train_steps(self.model_params, self.train_len)
        )
        # self.scheduler = get_constant_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=self.model_params["WARMUP_STEPS"]
        # )

        scheduler = self.scheduler
        self.opt = optimizer

        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=None, using_native_amp=False, using_lbfgs=None, second_order_closure=None): # ,
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.scheduler.step()
        self.log('Learning_rate', self.scheduler.get_last_lr()[0])

    def test_step(self, batch, batch_idx):
        input_ids = batch['source_ids']
        attention_mask = batch['source_mask']  # attention_mask
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=self.model_params["NUM_BEAMS"],
            min_length=self.model_params["MIN_TARGET_TEXT_LENGTH"],
            max_length=self.model_params["MAX_TARGET_TEXT_LENGTH"],
            repetition_penalty=2.0,  # 2.5,
            length_penalty=0.4,  # 1.0,
            early_stopping=False,  # True,
        )

        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]

        return {"preds": preds}

    def validation_step(self, batch, batch_idx):
        target_ids = batch['target_ids']
        target_text = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in target_ids]

        preds = self.test_step(batch, batch_idx)
        preds_text = preds["preds"]

        jaccard_score = [jaccard(p, t) for p, t in zip(preds_text, target_text)]
        rouge_scores = [rouge_score(p, t) for p, t in zip(preds_text, target_text)]

        avg_jaccard_score_batch = np.mean(jaccard_score)
        avg_rouge_scores_batch = np.mean(rouge_scores, 0)

        # log metrics
        self.log('jaccard_batch_val', avg_jaccard_score_batch)
        self.log('ROUGE1', avg_rouge_scores_batch[0])
        self.log('ROUGE2', avg_rouge_scores_batch[1])
        self.log('ROUGEL', avg_rouge_scores_batch[2])

        return {"jaccard_score": jaccard_score, "ROUGE1": avg_rouge_scores_batch[0], "ROUGE2": avg_rouge_scores_batch[1], "ROUGEL": avg_rouge_scores_batch[2]}

    def validation_end(self, outputs):
        jaccard_scores = sum([x["jaccard_score"] for x in outputs], [])
        avg_jaccard_score = np.mean(jaccard_scores)

        # self.log('val_jaccard', avg_jaccard_score)
        tensorboard_logs = {"jaccard_score": avg_jaccard_score}
        return {"avg_jaccard_score": avg_jaccard_score, "log": tensorboard_logs}

