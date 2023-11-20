import timm
import os
import logging
import torch
import requests
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


NO_MEAN = [0, 0, 0]
NO_STD = [1, 1, 1]

URL_LOOKUP = {
    'resnet50_trained_on_SIN':  'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
    'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
    'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    'resnet50_prime': 'https://zenodo.org/record/5801872/files/ResNet50_ImageNet_PRIME_noJSD.ckpt?download=1',
    'resnet50_moco_v3_100ep': 'https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/linear-100ep.pth.tar',
    'resnet50_moco_v3_300ep': 'https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/linear-300ep.pth.tar',
    'resnet50_moco_v3_1000ep': 'https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.pth.tar',
}

GID_LOOKUP = {
    'resnet50_pixmix_90ep':         '1_i45yvC88hos50QjkoD97OgbDGreKdp9',
    'resnet50_pixmix_180ep':        '1cgKYXDym3wgquf-4hwr_Ra3qje6GNHPH',
    'resnet50_augmix_180ep':        '1z-1V3rdFiwqSECz7Wkmn4VJVefJGJGiF',
    'resnet50_deepaugment':         '1DPRElQnBG66nd7GUphVm1t-5NroL7t7k',
    'resnet50_deepaugment_augmix':  '1QKmc_p6-qDkh51WvsaS9HKFv8bX5jLnP',
    'resnet50_noisymix':            '1Na79fzPZ0Azg01h6kGn1Xu5NoWOElSuG',
    'resnet50_hybridaugment_pp':    '1SpRU3oU3lZAuNDD-ncNkKN4Nbxnbfkoq'
}


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def load_state_dict_from_gdrive(id, model_name):
    state_path = os.path.join(torch.hub.get_dir(), f"{model_name}.pth")

    if not os.path.exists(state_path):
        logging.info(f"Downloading {id} to {state_path}")
        os.makedirs(torch.hub.get_dir(), exist_ok=True)
        download_file_from_google_drive(id, state_path)
    state = torch.load(state_path, map_location="cpu")
    return state


class NormalizedModel(torch.nn.Module):

    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.nn.Parameter(torch.Tensor(mean).view(-1, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.Tensor(std).view(-1, 1, 1), requires_grad=False)

    def forward(self, x):
        out = (x - self.mean) / self.std 
        out = self.model(out)
        return out


def get_normalized_model(model_name, eval=True):
    model = None
    # Good ol' AlexNet
    if model_name == "alexnet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        model = NormalizedModel(model, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # AT ResNets from Microsoft
    elif model_name.startswith("robust_resnet50"):
        model = timm.create_model("resnet50", pretrained=False)
        # get torch state from url
        tag = model_name.replace("robust_", "")
        state = torch.hub.load_state_dict_from_url(f"https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/{tag}.ckpt", map_location="cpu")["model"]
        state = {k.replace("module.model.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model = NormalizedModel(model, model.default_cfg.get("mean"), model.default_cfg.get("std"))
    # DINOv1 -> requires handling of seperated backbone/head
    elif model_name == "resnet50_dino":
        model = timm.create_model("resnet50", pretrained=False)
        # load backbone
        state = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state, strict=False)
        # load classifier head
        state = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_linearweights.pth",
            map_location="cpu",
        )["state_dict"]
        state = {k.replace("module.linear.", "fc."): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model = NormalizedModel(model, model.default_cfg.get("mean"), model.default_cfg.get("std"))
    elif model_name == "resnet50_swav":
        model = timm.create_model("resnet50", pretrained=False)
        # load backbone
        state = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
            map_location="cpu",
        )
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        # load classifier head
        state = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_eval_linear.pth.tar",
            map_location="cpu",
        )["state_dict"]
        state = {k.replace("module.linear.", "fc."): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model = NormalizedModel(model, model.default_cfg.get("mean"), model.default_cfg.get("std"))
    # Models accessible via http
    elif model_name in URL_LOOKUP:
        url = URL_LOOKUP.get(model_name)
        model = timm.create_model("resnet50", pretrained=False)
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model = NormalizedModel(model, model.default_cfg.get("mean"), model.default_cfg.get("std"))
    # Improved torchvision R50
    elif model_name == "tv2_resnet50":
        model = torch.hub.load("pytorch/vision", "resnet50", weights="ResNet50_Weights.IMAGENET1K_V2")
        model = NormalizedModel(model, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # SupCon
    elif model_name == "resnet50_supcon":
        model = timm.create_model("resnet50", pretrained=False)
        url = GID_LOOKUP.get(model_name)
        state = load_state_dict_from_gdrive(url, model_name)
        state = state["model"]
        state = {k.replace("module.", "").replace("encoder.", "").replace("head.2.", "fc."): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model = NormalizedModel(model, model.default_cfg.get("mean"), model.default_cfg.get("std"))
    # Models hosted on Google Drive
    elif model_name in GID_LOOKUP:
        model = timm.create_model("resnet50", pretrained=False)
        url = GID_LOOKUP.get(model_name)
        state = load_state_dict_from_gdrive(url, model_name)
        if "state_dict" in state.keys():
            state = state["state_dict"]
        if "model_state_dict" in state.keys():
            state = state["model_state_dict"]
        if "online_backbone" in state.keys():
            state = state["online_backbone"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model = NormalizedModel(model, model.default_cfg.get("mean"), model.default_cfg.get("std"))
    # Default to timm models
    else:
        pretrained = True
        if model_name.endswith("_untrained"):
            pretrained = False
            model_name = model_name.replace("_untrained", "")
        model = timm.create_model(model_name, pretrained=pretrained)
        model = NormalizedModel(model, model.default_cfg.get("mean"), model.default_cfg.get("std"))

    if eval:
        model.eval()
    return model