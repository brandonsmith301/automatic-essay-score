import math
import torch

def get_optimizer_params_with_llrd(model, encoder_lr, decoder_lr, weight_decay=0.0, learning_rate_llrd_mult=1.0):
    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    init_lr = encoder_lr
    head_lr = decoder_lr
    lr = init_lr
    print(f'Learning Rates: \n\tHead LR: {init_lr}')

    params_0 = [p for n, p in named_parameters if ("pooler" in n or "regressor" in n) and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "regressor" in n) and not any(nd in n for nd in no_decay)]

    optimizer_parameters.append({"params": params_0, "lr": head_lr, "weight_decay": 0.0})
    optimizer_parameters.append({"params": params_1, "lr": head_lr, "weight_decay": weight_decay})

    for layer in range(24, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n and any(nd in n for nd in no_decay)]
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n and not any(nd in n for nd in no_decay)]

        optimizer_parameters.append({"params": params_0, "lr": lr, "weight_decay": 0.0})
        optimizer_parameters.append({"params": params_1, "lr": lr, "weight_decay": weight_decay})
        
        print(f'\tLayer {layer} LR: {lr}')
        lr *= learning_rate_llrd_mult

    print(f'\tEmbeddings LR: {lr}')
    params_0 = [p for n, p in named_parameters if "embeddings" in n and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if "embeddings" in n and not any(nd in n for nd in no_decay)]

    optimizer_parameters.append({"params": params_0, "lr": lr, "weight_decay": 0.0})
    optimizer_parameters.append({"params": params_1, "lr": lr, "weight_decay": weight_decay})

    return optimizer_parameters


def deberta_base_adamw_grouped_llrd(model, encoder_lr, decoder_lr, init_weight_decay, factor):
    opt_parameters = []
    named_parameters = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    set_layers = [
        ["layer.0.", "layer.1.", "layer.2.", "layer.3."],
        ["layer.4.", "layer.5.", "layer.6.", "layer.7."],
        ["layer.8.", "layer.9.", "layer.10.", "layer.11."],
        ["layer.12.", "layer.13.", "layer.14.", "layer.15."],
        ["layer.16.", "layer.17.", "layer.18.", "layer.19."],
        ["layer.20.", "layer.21.", "layer.22.", "layer.23."]
    ]

    for name, params in named_parameters:
        weight_decay = 0.0 if any(p in name for p in no_decay) else init_weight_decay

        if name.startswith("backbone.embeddings") or name.startswith("backbone.encoder"):
            lr = encoder_lr
            for i, layer_set in enumerate(set_layers, 1):
                if any(layer in name for layer in layer_set):
                    lr = encoder_lr * (factor ** i)
                    break

            opt_parameters.append({"params": params, "weight_decay": weight_decay, "lr": lr})

        if name.startswith("fc") or name.startswith("backbone.pooler"):
            opt_parameters.append({"params": params, "weight_decay": weight_decay, "lr": decoder_lr})

    return opt_parameters


def get_parameters_groups(n_layers, n_groups):
    layers = [f'backbone.encoder.layer.{n_layers - i - 1}.' for i in range(n_layers)]
    step = math.ceil(n_layers / n_groups)
    groups = [layers[i:i + step] for i in range(0, n_layers, step)]
    return groups


def get_grouped_llrd_parameters(model, encoder_lr, decoder_lr, embeddings_lr, lr_mult_factor, weight_decay, n_groups):
    opt_parameters = []
    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    n_layers = model.backbone_config.num_hidden_layers
    parameters_groups = get_parameters_groups(n_layers, n_groups)

    for name, params in named_parameters:
        wd = 0.0 if any(p in name for p in no_decay) else weight_decay

        if name.startswith("backbone.encoder"):
            lr = encoder_lr
            for i, group in enumerate(parameters_groups, 1):
                if any(layer in name for layer in group):
                    lr = encoder_lr * (lr_mult_factor ** i)
                    break

            opt_parameters.append({"params": params, "weight_decay": wd, "lr": lr})

        if name.startswith("backbone.embeddings"):
            opt_parameters.append({"params": params, "weight_decay": wd, "lr": embeddings_lr})

        if name.startswith("fc") or name.startswith("backbone.pooler"):
            opt_parameters.append({"params": params, "weight_decay": wd, "lr": decoder_lr})

    return opt_parameters


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0
        },
        {
            'params': [p for n, p in model.named_parameters() if "backbone" not in n],
            'lr': decoder_lr, 'weight_decay': 0.0
        }
    ]
    return optimizer_parameters
