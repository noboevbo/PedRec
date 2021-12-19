import torch

def initialize_weights_with_same_name_and_shape(net, p2d_coco_only_weights_path: str, prefix: str = None):
    p2d_coco_only_weights = torch.load(p2d_coco_only_weights_path)
    net_weights = net.state_dict()
    for name, param in p2d_coco_only_weights.items():
        # if name == "loss_head.sigmas":
        #     # init p2d and p3d head sigmas
        #     net_weights[name][:param.shape[0]] = param[:]
        #     continue
        fixed_name = name.replace("conv_transpose_2d", "conv_transpose_shared")
        if prefix is not None:
            fixed_name = fixed_name.replace(prefix, "")
        if name in net_weights:
            net_weights[name] = param
        elif fixed_name in net_weights:
            net_weights[fixed_name] = param
        else:
            print(f"No weight found for '{name}'")
    net.load_state_dict(net_weights)


def initialize_p2d_from_p2d_coco(net, p2d_coco_only_weights_path: str):
    p2d_coco_only_weights = torch.load(p2d_coco_only_weights_path)
    net_weights = net.state_dict()
    for name, param in p2d_coco_only_weights.items():
        if name in net_weights:
            net_weights[name] = param
        else:
            print(f"No weight found for '{name}'")
    net.load_state_dict(net_weights)


def initialize_p3d_from_p2d_coco(net, p2d_coco_only_weights_path: str):
    p2d_coco_only_weights = torch.load(p2d_coco_only_weights_path)
    net_weights = net.state_dict()
    for name, param in p2d_coco_only_weights.items():
        net_name = name.replace("2d", "3d")
        if net_name in net_weights:
            net_weights[net_name] = param
        else:
            print(f"No weight found for '{name}', tried mapping '{net_name}'")
    net.load_state_dict(net_weights)


def initialize_p2d_p3d_from_p2d_coco(net, p2d_coco_only_weights_path: str):
    p2d_coco_only_weights = torch.load(p2d_coco_only_weights_path)
    net_weights = net.state_dict()
    for name, param in p2d_coco_only_weights.items():
        if name in net_weights:
            net_weights[name] = param
        # try to initialize 3d from 2d
        net_name = name.replace("2d", "3d")
        if net_name in net_weights:
            net_weights[net_name] = param
        else:
            print(f"No weight found for '{name}', tried mapping '{net_name}'")
    net.load_state_dict(net_weights)


def initialize_p2d_p3d_shared_conv_from_p2d_coco(net, p2d_coco_only_weights_path: str):
    p2d_coco_only_weights = torch.load(p2d_coco_only_weights_path)
    net_weights = net.state_dict()
    for name, param in p2d_coco_only_weights.items():
        if name in net_weights:
            net_weights[name] = param
        else:
            # try to initialize 3d from 2d
            net_name = name.replace("2d", "shared")
            if net_name in net_weights:
                net_weights[net_name] = param
            else:
                net_name = name.replace("shared", "3d")
                if net_name in net_weights:
                    net_weights[net_name] = param
                else:
                    print(f"No weight found for '{name}', tried mapping '{net_name}'")
    net.load_state_dict(net_weights)