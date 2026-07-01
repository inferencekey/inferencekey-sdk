### Vendored (inference-only subset) from LP-MusicCaps
### (seungheondoh/lp-music-caps). ``load_pretrained`` is ADAPTED from the
### original: the forced ``.cuda()`` is removed and the target device is passed
### in, so the model can run on CPU. Training-only helpers are omitted.

import torch


def load_pretrained(weights_path, model, device="cpu", mdp=False):
    """Load LP-MusicCaps transfer weights into ``model`` and move it to ``device``.

    Args:
        weights_path: path to the ``transfer.pth`` checkpoint.
        model: an instantiated ``BartCaptionModel``.
        device: torch device string (e.g. ``"cpu"`` or ``"cuda:0"``).
        mdp: whether the checkpoint keys carry a ``module.`` prefix (from
            multiprocessing-distributed training). ``transfer.pth`` does not, so
            this defaults to ``False``.

    Returns:
        (model, save_epoch)
    """
    pretrained_object = torch.load(weights_path, map_location="cpu")
    state_dict = pretrained_object["state_dict"]
    save_epoch = pretrained_object.get("epoch")
    if mdp:
        for k in list(state_dict.keys()):
            if k.startswith("module."):
                state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, save_epoch
