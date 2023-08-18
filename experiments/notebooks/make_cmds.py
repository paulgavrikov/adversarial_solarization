import itertools

CMD = "conda run -n convprops python /workspace/gavrikov/adversarial_solarization/experiments/notebooks/randomized_solarization_attack_imagenet.py"

config = {
    "model": ["alexnet", "resnet18", "vgg16_bn", "tv_densenet121", "resnet34", "tv_resnet50",
             "repvgg_b1", "tf_efficientnet_b0", "convmixer_768_32", "resnet50", "resnet101", 
             "convnext_base", "vit_base_patch16_224", "robust_resnet50_linf_eps4", 
             "resnetv2_50x1_bit_distilled", "resnet50_noisymix"],
    "target": ["top1", "top5"],
    "iterations": [10]
}


class dotdict(dict):  # https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


keys = list(config.keys())
values = map(lambda x: config[x], keys)

i = 0
for values_tuple in itertools.product(*values):
    e = dotdict(dict(zip(keys, values_tuple)))

    cmd = f"{CMD} --imagenet /workspace/data/datasets/imagenet --model {e.model} --target {e.target} --iterations {e.iterations} --device cuda:%gpu%"
    print(cmd)
    i+=1 
    
print(f"# {i} commands")
