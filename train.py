import warnings
warnings.filterwarnings("ignore")

import os, sys, time, random, gc, argparse, wandb 
os.environ["TOKENIZERS_PARALLELISM"] = "true"
sys.path.append('../src')

from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint 
from transformers import AutoTokenizer
from tokenizers import AddedToken
from sklearn.model_selection import StratifiedKFold

from utils import (
    load_filepaths,
    get_config,
    dictionary_to_namespace,
    get_logger,
    save_config,
    update_filepaths,
    AverageMeter,
    time_since,
    get_evaluation_steps,
    str_to_bool,
    create_dirs_if_not_exists
)

from dataset.preprocess import (
    make_folds,
    get_max_len_from_df,
    get_additional_special_tokens,
    preprocess_text
)

from criterion.metric import get_score
from dataset.datasets import get_train_dataloader, get_valid_dataloader, collate
from model.utils import get_model
from optimizer.optimizer import get_optimizer
from scheduler.scheduler import get_scheduler
from adversarial_learning.awp import AWP
from criterion.criterion import get_criterion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--fold', type=int)
    arguments = parser.parse_args()
    return arguments


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def valid_fn(valid_dataloader, model, criterion, epoch):
    valid_losses = AverageMeter()
    model.eval()
    predictions = []
    start = time.time()

    for step, (inputs, labels) in enumerate(valid_dataloader):
        inputs = collate(inputs)

        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)

        if config.training.gradient_accumulation_steps > 1:
            loss = loss / config.training.gradient_accumulation_steps

        valid_losses.update(loss.item(), batch_size)
        predictions.append(y_preds.to('cpu').numpy())

        if step % config.general.valid_print_frequency == 0 or step == (len(valid_dataloader) - 1):
            remain = time_since(start, float(step + 1) / len(valid_dataloader))
            logger.info('EVAL: [{0}][{1}/{2}] '
                        'Elapsed: {remain:s} '
                        'Loss: {loss.avg:.4f} '
                        .format(epoch+1, step+1, len(valid_dataloader),
                                remain=remain,
                                loss=valid_losses))


    predictions = np.concatenate(predictions)
    return valid_losses, predictions


def inference_fn(test_loader, model):
    predictions = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        predictions.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(predictions)
    return 0, predictions


def train_loop(
    train_folds,
    valid_folds,
    model_checkpoint_path=None
):
    train_dataloader = get_train_dataloader(config, train_folds)
    valid_dataloader = get_valid_dataloader(config, valid_folds)
    valid_labels = valid_folds[config.general.target_columns].values

    model = get_model(config, model_checkpoint_path=model_checkpoint_path)
    torch.save(model.backbone_config, filepaths['backbone_config_fn_path'])
    model.to(device)
    
    optimizer = get_optimizer(model, config)
    train_steps_per_epoch = int(len(train_folds) / config.general.train_batch_size)
    num_train_steps = train_steps_per_epoch * config.training.epochs

    eval_steps = get_evaluation_steps(
        train_steps_per_epoch,
        config.training.evaluate_n_times_per_epoch
    )

    scheduler = get_scheduler(optimizer, config, num_train_steps)
    awp = AWP(
        model=model,
        optimizer=optimizer,
        adv_lr=config.adversarial_learning.adversarial_lr,
        adv_eps=config.adversarial_learning.adversarial_eps,
        adv_epoch=config.adversarial_learning.adversarial_epoch_start
    )
    
    criterion = get_criterion(config)
    best_score = -np.inf
    
    for epoch in range(config.training.epochs):
        start_time = time.time()
        model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=config.training.apex)
        
        train_losses = AverageMeter()
        valid_losses = None
        score = None
        start = time.time()
        global_step = 0

        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(device)
                
            labels = labels.to(device)
            awp.perturb(epoch)
            batch_size = labels.size(0)

            with torch.cuda.amp.autocast(enabled=config.training.apex):
                y_preds = model(inputs)
                loss = criterion(y_preds, labels)

            if config.training.gradient_accumulation_steps > 1:
                loss = loss / config.training.gradient_accumulation_steps

            train_losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            awp.restore()

            if config.training.unscale:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.training.max_grad_norm
            )

            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                if config.scheduler.batch_scheduler:
                    scheduler.step()

            if (step % config.general.train_print_frequency == 0) or \
                    (step == (len(train_dataloader) - 1)) or \
                    (step + 1 in eval_steps) or \
                    (step - 1 in eval_steps):

                remain = time_since(start, float(step + 1) / len(train_dataloader))
                logger.info(f'Epoch: [{epoch+1}][{step+1}/{len(train_dataloader)}] '
                            f'Elapsed {remain:s} '
                            f'Loss: {train_losses.val:.4f}({train_losses.avg:.4f}) '
                            f'Grad: {grad_norm:.4f}  '
                            f'LR: {scheduler.get_lr()[0]:.8f}  ')

            if (step + 1) in eval_steps:
                valid_losses, predictions = valid_fn(valid_dataloader, model, criterion, epoch)
                score = get_score(valid_labels, predictions)

                model.train()
                logger.info(f'Epoch {epoch+1} - Score: {score:.4f}')
                if score > best_score:
                    best_score = score
                    torch.save({'model': model.state_dict(), 'predictions': predictions}, filepaths['model_fn_path'])
                    logger.info(f'\nEpoch {epoch + 1} - Save Best Score: {best_score:.4f} Model\n')

                unique_parameters = ['.'.join(name.split('.')[:4]) for name, _ in model.named_parameters()]
                learning_rates = list(set(zip(unique_parameters, scheduler.get_lr())))

        if config.optimizer.use_swa:
            optimizer.swap_swa_sgd()

        elapsed = time.time() - start_time

        logger.info(f'Epoch {epoch + 1} - avg_train_loss: {train_losses.avg:.4f} '
                    f'avg_val_loss: {valid_losses.avg:.4f} time: {elapsed:.0f}s '
                    f'Epoch {epoch + 1} - Score: {score:.4f}\n'
                    '=============================================================================\n')

    predictions = torch.load(filepaths['model_fn_path'], map_location=torch.device('cpu'))['predictions']
    valid_folds["pred"] = predictions
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds


def get_result(oof_df):
    labels = oof_df[config.general.target_columns].values
    preds = oof_df["pred"].values
    score = get_score(labels, preds)
    print(f'Score: {score:<.4f}')


def check_arguments():
    all_folds = [i for i in range(config.general.n_folds)]
    assert args.fold in all_folds, \
        f'Invalid training fold, fold number must be in {all_folds}'
    

def main():
    train = pd.read_csv("training_datasets/annotated_dataset.csv")
    train['full_text'] = train['full_text'].apply(preprocess_text)
    
    extra = train[train['type']!='comp'].reset_index(drop=True)
    train = train[train['type']=='comp'].reset_index(drop=True)
    
    train = make_folds(
        train,
        target_cols=config.general.target_columns,
        n_splits=config.general.n_folds,
        random_state=config.general.seed
    )

    
    special_tokens_replacement = get_additional_special_tokens()
    all_special_tokens = list(special_tokens_replacement.values())

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.backbone_type,
        use_fast=True,
        additional_special_tokens=all_special_tokens
    )
    
    tokenizer.add_tokens([AddedToken(" "*2, normalized=False)])
    tokenizer.add_tokens([AddedToken("[SPELL_ERROR]", normalized=False)])
    tokenizer.save_pretrained(filepaths['tokenizer_dir_path'])
    config.tokenizer = tokenizer

    train_df = train[train['fold'] != fold]
    train_df = pd.concat([train_df, extra], axis=0).reset_index(drop=True)
    valid_df = train[train['fold'] == fold].reset_index(drop=True)

    fold_out = train_loop(
        train_df,
        valid_df,
        model_checkpoint_path=None
    )
    fold_out.to_csv(filepaths['oof_fn_path'], index=False)


if __name__ == '__main__':
    args = parse_args()
    filepaths = load_filepaths()

    config_path = os.path.join(filepaths['CONFIGS_DIR_PATH'], args.config_name)
    config = get_config(config_path)
    fold = args.fold

    filepaths = update_filepaths(filepaths, config, args.run_id, fold)
    create_dirs_if_not_exists(filepaths)
    
    if not os.path.exists(filepaths['run_dir_path']):
        os.makedirs(filepaths['run_dir_path'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger(filename=filepaths['log_fn_path'])

    if os.path.isfile(filepaths['model_fn_path']):
        new_name = filepaths["model_fn_path"]+f'_renamed_at_{str(datetime.now())}'
        logger.warning(f'{filepaths["model_fn_path"]} is already exists, renaming this file to {new_name}')
        os.rename(filepaths["model_fn_path"], new_name)

    config = dictionary_to_namespace(config)
    seed_everything(seed=config.general.seed)
    check_arguments()
    main()