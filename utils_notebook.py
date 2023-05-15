import torch
import numpy as np
from typing import Dict, Optional, List
from dataloader import get_dataloaders
from models.msdnet_ge import MSDNet
from models.msdnet_imta import IMTA_MSDNet
from utils import parse_args
from collections import OrderedDict
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os


def probs_decrease(probs: np.array) -> np.array:
    L = len(probs)
    diffs = []
    for i in range(L):
        for j in range(i + 1, L):
            diffs.append(probs[j] - probs[i])
    return np.array(diffs)


def modal_probs_decreasing(
    _preds: Dict[int, torch.Tensor],
    _probs: torch.Tensor,
    layer: Optional[int] = None,
    verbose: bool = False,
    N: int = 10000,
    diffs_type: str = "consecutive",
    thresholds: List[float] = [-0.01, -0.05, -0.1, -0.2, -0.5],
) -> Dict[float, float]:
    """
    nr. of decreasing modal probability vectors in anytime-prediction regime

    function can also be used for grount truth probabilities, set layer=None
    """
    nr_non_decreasing = {threshold: 0 for threshold in thresholds}
    # diffs = []
    for i in range(N):
        if layer is None:
            c = _preds[i]
        else:
            c = _preds[layer - 1][i]
        probs_i = _probs[:, i, c].cpu().numpy()
        if diffs_type == "consecutive":
            diffs_i = np.diff(probs_i)
        elif diffs_type == "all":
            diffs_i = probs_decrease(probs_i)
        else:
            raise ValueError()
        # diffs.append(diffs_i.min())
        for threshold in nr_non_decreasing.keys():
            if np.all(diffs_i >= threshold):
                nr_non_decreasing[threshold] += 1
            else:
                if verbose:
                    print(i, probs_i)
    # print(nr_non_decreasing)
    # print(np.mean(diffs))
    nr_decreasing = {
        -1.0 * k: ((N - v) / N) * 100 for k, v in nr_non_decreasing.items()
    }
    return nr_decreasing


def f_probs_ovr_poe_logits_weighted(logits, threshold=0.):
    C = logits.shape[-1]
    probs = logits.numpy().copy()
    probs[probs < threshold] = 0.
    probs = np.cumprod(probs, axis=0)
    # normalize
    probs = (probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2))
    return probs


def f_probs_ovr_poe_logits_weighted_generalized(logits, threshold=0.0, weights=None):
    L, C = logits.shape[0], logits.shape[-1]
    probs = logits.numpy().copy()
    probs[probs < threshold] = 0.0
    if weights is not None:
        assert logits.shape[0] == weights.shape[0]
        for l in range(L):
            probs[l, :, :] = probs[l, :, :] ** weights[l]
    probs = np.cumprod(probs, axis=0)
    # normalize
    probs = probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2)
    return probs


def anytime_caching(_probs: torch.Tensor, N: int, L: int) -> torch.Tensor:
    _preds = []
    _probs_stateful = []
    for n in range(N):
        preds_all, probs_stateful_all = [], []
        max_prob_all, pred_all, max_id = 0., None, 0.
        for l in range(L):
            _max_prob, _pred = _probs[l, n, :].max(), _probs[l, n, :].argmax()
            if _max_prob >= max_prob_all:
                max_prob_all = _max_prob
                pred_all = _pred
                prob_stateful_all = _probs[l, n, :]
                max_id = l
            else:
                prob_stateful_all = _probs[max_id, n, :]
            preds_all.append(pred_all)
            probs_stateful_all.append(prob_stateful_all)
        _preds.append(torch.stack(preds_all))
        _probs_stateful.append(torch.stack(probs_stateful_all))

    _preds = torch.stack(_preds)
    _probs_stateful = torch.stack(_probs_stateful)
    return _probs_stateful.permute(1, 0, 2)


def get_metrics_for_paper(logits: torch.Tensor, targets: torch.Tensor, model_name: str, thresholds: List[float] = [-0.0001, -0.01, -0.05, -0.1, -0.2, -0.25, -0.33, -0.5]):

    L = len(logits)
    N = len(targets)

    acc_dict, mono_modal_dict, mono_ground_truth_dict = {}, {}, {}

    probs = torch.softmax(logits, dim=2)
    preds = {i: torch.argmax(probs, dim=2)[i, :] for i in range(L)}
    acc = [(targets == preds[i]).sum() / len(targets) for i in range(L)]

    probs_pa = torch.tensor(f_probs_ovr_poe_logits_weighted_generalized(logits, weights=(np.arange(1, L + 1, 1, dtype=float) / L)))
    preds_pa = {i: torch.argmax(probs_pa, dim=2)[i, :] for i in range(L)}
    acc_pa = [(targets == preds_pa[i]).sum() / len(targets) for i in range(L)]

    probs_ca = anytime_caching(probs, N=N, L=L)
    preds_ca= {i: torch.argmax(probs_ca, dim=2)[i, :] for i in range(L)}
    acc_ca = [(targets == preds_ca[i]).sum() / len(targets) for i in range(L)]

    for _probs, _preds, _acc, _name in zip([probs, probs_pa, probs_ca], [preds, preds_pa, preds_ca], [acc, acc_pa, acc_ca], [model_name, model_name + '-PA', model_name + '-CA']):
        acc_dict[_name] = [round(float(x), 4) for x in _acc]
        mono_modal_dict[_name] = [round(x, 4) for x in modal_probs_decreasing(_preds, _probs, layer=L, N=N, thresholds=thresholds, diffs_type="all").values()]
        mono_ground_truth_dict[_name] = [round(x, 4) for x in modal_probs_decreasing(targets, _probs, layer=None, N=N, thresholds=thresholds, diffs_type="all").values()]

    return acc_dict, mono_modal_dict, mono_ground_truth_dict



def get_logits_targets_imta(model_name: str, dataset: str, epoch: int, model_pretrained: str, epoch_pretrained: int):
    ARGS = parse_args()
    ARGS.data_root = 'data'
    ARGS.data = dataset
    ARGS.save= f'/home/metod/Desktop/PhD/year1/PoE/IMTA/_models/{ARGS.data}/{model_name}'
    ARGS.arch = 'IMTA_MSDNet'
    ARGS.grFactor = [1, 2, 4]
    ARGS.bnFactor = [1, 2, 4]
    ARGS.growthRate = 6
    ARGS.batch_size = 64
    ARGS.epochs = 300
    ARGS.nBlocks = 7
    ARGS.stepmode = 'even'
    ARGS.base = 4
    ARGS.nChannels = 16
    if ARGS.data == 'cifar10':
        ARGS.num_classes = 10
    elif ARGS.data == 'cifar100':
        ARGS.num_classes = 100
    else:
        raise ValueError('Unknown dataset')
    ARGS.step = 2
    ARGS.use_valid = True
    ARGS.splits = ['train', 'val', 'test']
    ARGS.nScales = len(ARGS.grFactor)

    ARGS.T = 1.0
    ARGS.gamma = 0.1
    ARGS.pretrained = f'/home/metod/Desktop/PhD/year1/PoE/IMTA/_models/{ARGS.data}/{model_pretrained}/save_models/checkpoint_{epoch_pretrained}.pth.tar'

    problematic_prefix = 'module.'

    # load pre-trained model
    model = IMTA_MSDNet(args=ARGS)
    MODEL_PATH = f'_models/{ARGS.data}/{model_name}/save_models/checkpoint_{epoch}.pth.tar'
    # MODEL_PATH = f'_models/{ARGS.data}/{MODEL}/save_models/model_best.pth.tar'  # TODO: investigate why using this results in poor accuracy of baseline model
    print(MODEL_PATH)
    state = torch.load(MODEL_PATH)
    params = OrderedDict()
    for params_name, params_val in state['state_dict'].items():
        if params_name.startswith(problematic_prefix):
            params_name = params_name[len(problematic_prefix):]
        params[params_name] = params_val
    model.load_state_dict(params)
    model = model.cuda()
    model.eval()

    # data
    _, _, test_loader = get_dataloaders(ARGS)

    logits = []
    targets = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            y = y.cuda(device=None)
            x = x.cuda()

            input_var = torch.autograd.Variable(x)
            target_var = torch.autograd.Variable(y)

            output = model(input_var)

            if not isinstance(output, list):
                output = [output]

            logits.append(torch.stack(output))
            targets.append(target_var)

    logits = torch.cat(logits, dim=1).cpu()
    targets = torch.cat(targets).cpu()

    return logits, targets



def merge_dicts(data):
    merged = {}
    for d in data:
        for key, value in d.items():
            if key not in merged:
                merged[key] = []
            merged[key].append(value)
    
    result = {}
    for key, value_lists in merged.items():
        avg_values = [np.mean(values).round(4) for values in zip(*value_lists)]
        std_values = [np.std(values).round(4) for values in zip(*value_lists)]
        result[key] = (avg_values, std_values)
    
    return result



def get_logits_targets_imta_image_net(model_name, epoch, model_name_pretrained, epoch_pretrained, dataset='ImageNet'):
    ARGS = parse_args()
    ARGS.data_root = '/home/metod/Desktop/PhD/year1/PoE/MSDNet-PyTorch/data/image_net'
    ARGS.data = dataset
    ARGS.save= f'/home/metod/Desktop/PhD/year1/PoE/IMTA/_models/{ARGS.data}/{model_name}'
    ARGS.arch = 'IMTA_MSDNet'
    ARGS.grFactor = [1, 2, 4, 4]
    ARGS.bnFactor = [1, 2, 4, 4]
    ARGS.growthRate = 16
    ARGS.batch_size = 350
    ARGS.epochs = 90
    ARGS.nBlocks = 5
    ARGS.stepmode = 'even'
    ARGS.base = 4
    ARGS.nChannels = 32
    if ARGS.data == 'cifar10':
        ARGS.num_classes = 10
    elif ARGS.data == 'cifar100':
        ARGS.num_classes = 100
    elif ARGS.data == 'ImageNet':
        ARGS.num_classes = 1000
    else:
        raise ValueError('Unknown dataset')
    ARGS.step = 4
    ARGS.use_valid = True
    ARGS.splits = ['train', 'val', 'test']
    ARGS.nScales = len(ARGS.grFactor)

    
    ARGS.T = 1.0
    ARGS.gamma = 0.1
    ARGS.pretrained = f'/home/metod/Desktop/PhD/year1/PoE/IMTA/_models/{ARGS.data}/{model_name_pretrained}/checkpoint_0{epoch_pretrained}.pth.tar'

    problematic_prefix = 'module.'

    # load pre-trained model
    model = IMTA_MSDNet(args=ARGS)
  
    MODEL_PATH = f'_models/{ARGS.data}/{model_name}/checkpoint_0{epoch}.pth.tar'
    # MODEL_PATH = f'_models/{ARGS.data}/{MODEL}/save_models/model_best.pth.tar'  # TODO: investigate why using this results in poor accuracy of baseline model
    print(MODEL_PATH)
    state = torch.load(MODEL_PATH)
    params = OrderedDict()
    for params_name, params_val in state['state_dict'].items():
        if params_name.startswith(problematic_prefix):
            params_name = params_name[len(problematic_prefix):]
        params[params_name] = params_val
    model.load_state_dict(params)
    model = model.cuda()
    model.eval()

    valdir = os.path.join(ARGS.data_root, 'valid')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    val_set = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]))

    val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=ARGS.batch_size, shuffle=False,
                num_workers=ARGS.workers, pin_memory=True)
    
    logits = []
    targets = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            y = y.cuda(device=None)
            x = x.cuda()

            input_var = torch.autograd.Variable(x)
            target_var = torch.autograd.Variable(y)

            output = model(input_var)

            if not isinstance(output, list):
                output = [output]

            logits.append(torch.stack(output))
            targets.append(target_var)

    logits = torch.cat(logits, dim=1).cpu()
    targets = torch.cat(targets).cpu()

    return logits, targets



def get_metrics_with_error_bars(model_name: str, dataset: str, model_list: List):
    assert dataset in ['cifar10', 'cifar100', 'ImageNet']
    acc_res, mono_modal_res, mono_correct_res = [], [], []
    for _model, _epoch, _model_pretrained, _epoch_pretrained in model_list:
        if dataset != 'ImageNet':
            logits, targets = get_logits_targets_imta(model_name=_model, epoch=_epoch, dataset=dataset, 
                                                         model_pretrained=_model_pretrained, epoch_pretrained=_epoch_pretrained)
        else:
            logits, targets = get_logits_targets_imta_image_net(model_name=_model, epoch=_epoch, dataset=dataset, 
                                                         model_name_pretrained=_model_pretrained, epoch_pretrained=_epoch_pretrained)
        acc, mono_modal, mono_correct = get_metrics_for_paper(logits, targets, model_name=model_name)
        acc_res.append(acc)
        mono_modal_res.append(mono_modal)
        mono_correct_res.append(mono_correct)

    acc_res = merge_dicts(acc_res)
    mono_modal_res = merge_dicts(mono_modal_res)
    mono_correct_res = merge_dicts(mono_correct_res)

    return acc_res, mono_modal_res, mono_correct_res


