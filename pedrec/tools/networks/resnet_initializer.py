import torch
import torchvision.models as models


def initialize_diss_experiment_from_p2d_coco_only(net, p2d_coco_only_weights_path: str):
    p2d_coco_only_weights = torch.load(p2d_coco_only_weights_path)
    net_weights = net.state_dict()
    for name, param in p2d_coco_only_weights.items():
        if name.startswith("model.head_pose_3d") \
                or name.startswith("model.head_orientation") \
                or name.startswith("model.head_conf") \
                or name.startswith("model.head_orientation"):
            print(f"Skipped '{name}', was frozen in coco only p2d.")
            continue
        if name in net_weights:
            net_weights[name] = param
        else:
            print("(ERROR NOT FOUND) " + name)
    net.load_state_dict(net_weights)


def initialize_feature_extractor(net):
    resnet50 = models.resnet50(pretrained=True)
    net_weights = net.state_dict()
    for name, param in resnet50.state_dict().items():
        net_name = f"feature_extractor.{name}"
        if net_name in net_weights:
            net_weights[net_name] = param
    net.load_state_dict(net_weights)


def initialize_v5_mtl_from_noorientation(net, weights_path: str):
    weights = torch.load(weights_path)
    net_weights = net.state_dict()
    for name, param in net.state_dict().items():
        if name in weights and net_weights[name].shape == weights[name].shape:
            net_weights[name] = weights[name]
        if name == "loss_head.sigmas":
            net_weights[name][:2] = weights[name][:2]
            net_weights[name][3] = weights[name][2]
        # elif mapped_pose_3d_head_name in weights:
        #     if net_weights[name].shape == weights[mapped_pose_3d_head_name].shape:
        #         print(f"Init {name} deconv from Pose2D deconv")
        #         net_weights[name] = weights[mapped_pose_3d_head_name]
        #     else:
        #         # a = 1
        #         print(
        #             f"Found match for {name} with mapping: {mapped_pose_3d_head_name}, but shape a ({net_weights[name].shape}) differs from b ({weights[mapped_pose_3d_head_name].shape})")
        # else:
        #     # a = 1
        #     print(f"Skipped {name}, tried mapping: {mapped_pose_3d_head_name}")
    net.load_state_dict(net_weights)


def initialize_pose_resnet(net, pose_resnet_weights_path: str):
    pose_resnet_state_dict = torch.load(pose_resnet_weights_path)
    net_weights = net.state_dict()
    for name, param in pose_resnet_state_dict.items():
        if name.startswith("final"):
            net_name = name.replace("final_layer.", "head.pose_heatmap_layer.")
            # net_name = f"head.pose_3d_heatmap_layer.{name}"
        elif name.startswith("deconv"):
            net_name = f"head.{name}"
        else:
            net_name = f"feature_extractor.{name}"
        if net_name in net_weights:
            net_weights[net_name] = param
        else:
            print("Skipped:" + name)
    net.load_state_dict(net_weights)


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


def initialize_from_coco_trained_v3mtl(net, weights_path: str):
    weights = torch.load(weights_path)
    net_weights = net.state_dict()
    for name, param in net.state_dict().items():
        # print(name)
        mapped_pose_3d_head_name = f"model.{name}"
        if name in weights and net_weights[name].shape == weights[name].shape:
            net_weights[name] = weights[name]
        elif mapped_pose_3d_head_name in weights:
            if net_weights[name].shape == weights[mapped_pose_3d_head_name].shape:
                print(f"Init {name} deconv from Pose2D deconv")
                net_weights[name] = weights[mapped_pose_3d_head_name]
            else:
                # a = 1
                print(
                    f"Found match for {name} with mapping: {mapped_pose_3d_head_name}, but shape a ({net_weights[name].shape}) differs from b ({weights[mapped_pose_3d_head_name].shape})")
        else:
            # a = 1
            print(f"Skipped {name}, tried mapping: {mapped_pose_3d_head_name}")
    net.load_state_dict(net_weights)


def initialize_from_coco_trained_v2(net, weights_path: str):
    weights = torch.load(weights_path)
    net_weights = net.state_dict()
    for name, param in net.state_dict().items():
        # print(name)
        mapped_pose_3d_head_name = name.replace("deconv_head.0", "deconv_layers.6") \
            .replace("deconv_head.1", "deconv_layers.7") \
            .replace("deconv_head_2d.0", "deconv_layers.6") \
            .replace("deconv_head_2d.1", "deconv_layers.7") \
            .replace("deconv_head_3d.0", "deconv_layers.6") \
            .replace("deconv_head_3d.1", "deconv_layers.7") \
            .replace("head_pose_2d.", "head.") \
            .replace("conv_transpose_shared_2.", "head.") \
            .replace("conv_transpose_shared.", "head.") \
            .replace("head_pose_3d.", "head.") \
            .replace("head_pose.", "head.") \
            .replace("head_orientation.", "head.") \
            .replace("pose_2d_heatmap_layer.", "pose_heatmap_layer.") \
            .replace("pose_3d_heatmap_layer.", "pose_heatmap_layer.")
        if name.startswith("deconv_layers."):
            mapped_pose_3d_head_name = f"head.{name}"  # Append head because coco only deconv was in head.
        mapped_pose_3d_head_name = mapped_pose_3d_head_name.replace("deconv_head_2d", "deconv_layers") \
            .replace("deconv_head_3d", "deconv_layers") \
            .replace("deconv_head", "deconv_layers")
        if name in weights and net_weights[name].shape == weights[name].shape:
            net_weights[name] = weights[name]
        elif mapped_pose_3d_head_name in weights:
            if net_weights[name].shape == weights[mapped_pose_3d_head_name].shape:
                print(f"Init {name} deconv from Pose2D deconv")
                net_weights[name] = weights[mapped_pose_3d_head_name]
            else:
                if name == "head_pose_2d.pose_heatmap_layer.weight" or name == "head_pose_2d.pose_heatmap_layer.bias":
                    net_weights[name][0:17] = weights[mapped_pose_3d_head_name]
                    print("mapped head_pose_2d with more joints from lower joints")
                else:
                    print(
                        f"Found match for {name} with mapping: {mapped_pose_3d_head_name}, but shape a ({net_weights[name].shape}) differs from b ({weights[mapped_pose_3d_head_name].shape})")
        else:
            # a = 1
            print(f"Skipped {name}, tried mapping: {mapped_pose_3d_head_name}")
    net.load_state_dict(net_weights)


def initialize_from_p2d(net, weights_path: str):
    weights = torch.load(weights_path)
    net_weights = net.state_dict()
    for name, param in net.state_dict().items():
        # print(name)
        mapped_pose_3d_head_name = name.replace("head_pose_3d.", "head_pose_2d.") \
            .replace("head_orientation.", "head_pose_2d.") \
            .replace("pose_3d_heatmap_layer.", "pose_heatmap_layer.")
        mapped_pose_3d_head_name = f"model.{mapped_pose_3d_head_name}"
        if name in weights and net_weights[name].shape == weights[name].shape:
            net_weights[name] = weights[name]
        elif mapped_pose_3d_head_name in weights:
            if net_weights[name].shape == weights[mapped_pose_3d_head_name].shape:
                print(f"Init {name} deconv from Pose2D deconv")
                net_weights[name] = weights[mapped_pose_3d_head_name]
            else:
                if name == "head_pose_2d.pose_heatmap_layer.weight" or name == "head_pose_2d.pose_heatmap_layer.bias":
                    net_weights[name][0:17] = weights[mapped_pose_3d_head_name]
                    print("mapped head_pose_2d with more joints from lower joints")
                else:
                    print(
                        f"Found match for {name} with mapping: {mapped_pose_3d_head_name}, but shape a ({net_weights[name].shape}) differs from b ({weights[mapped_pose_3d_head_name].shape})")
        else:
            # a = 1
            print(f"Skipped {name}, tried mapping: {mapped_pose_3d_head_name}")
    net.load_state_dict(net_weights)


def initialize_from_p2d_p3d(net, weights_path: str):
    weights = torch.load(weights_path)
    net_weights = net.state_dict()
    for name, param in net.state_dict().items():
        # print(name)
        mapped_pose_3d_head_name = name.replace("head_orientation.", "head_pose_2d.") \
            # .replace("pose_3d_heatmap_layer.", "pose_heatmap_layer.")
        mapped_pose_3d_head_name = f"model.{mapped_pose_3d_head_name}"
        if name in weights and net_weights[name].shape == weights[name].shape:
            net_weights[name] = weights[name]
        elif mapped_pose_3d_head_name in weights:
            if net_weights[name].shape == weights[mapped_pose_3d_head_name].shape:
                print(f"Init {name} deconv from Pose2D deconv")
                net_weights[name] = weights[mapped_pose_3d_head_name]
            else:
                if name == "head_pose_2d.pose_heatmap_layer.weight" or name == "head_pose_2d.pose_heatmap_layer.bias":
                    net_weights[name][0:17] = weights[mapped_pose_3d_head_name]
                    print("mapped head_pose_2d with more joints from lower joints")
                else:
                    print(
                        f"Found match for {name} with mapping: {mapped_pose_3d_head_name}, but shape a ({net_weights[name].shape}) differs from b ({weights[mapped_pose_3d_head_name].shape})")
        else:
            # a = 1
            print(f"Skipped {name}, tried mapping: {mapped_pose_3d_head_name}")
    net.load_state_dict(net_weights)


def initialize_from_coco_trained(net, weights_path: str):
    weights = torch.load(weights_path)
    net_weights = net.state_dict()
    for name, param in net.state_dict().items():
        print(name)
        mapped_pose_3d_head_name = name.replace("head_pose_3d.", "head.")
        mapped_pose_3d_head_name = mapped_pose_3d_head_name.replace("pose_3d_heatmap_layer.", "pose_heatmap_layer.")
        if name.startswith("deconv_layers."):
            mapped_pose_3d_head_name = f"head.{name}"  # Append head because coco only deconv was in head.
        mapped_pose_3d_head_name = mapped_pose_3d_head_name.replace("deconv_head", "deconv_layers")
        if name in weights and net_weights[name].shape == weights[name].shape:
            net_weights[name] = weights[name]
        elif mapped_pose_3d_head_name in weights:
            if net_weights[name].shape == weights[mapped_pose_3d_head_name].shape:
                print(f"Init Pose3D deconv from Pose2D deconv")
                net_weights[name] = weights[mapped_pose_3d_head_name]
            else:
                print(
                    f"Found match for {name} with mapping: {mapped_pose_3d_head_name}, but shape a ({net_weights[name].shape}) differs from b ({weights[mapped_pose_3d_head_name].shape})")
        else:
            print(f"Skipped {name}, tried mapping: {mapped_pose_3d_head_name}")
    net.load_state_dict(net_weights)
