"""Small tensor helpers used by the reconstruction model."""
def ImgRenormalize(images):
    return (images + 1.0) / 2.0

def freeze_all_params(modules):
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    for module in modules:
        module.requires_grad_(False)
