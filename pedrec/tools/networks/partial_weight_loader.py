import torch

from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.models.data_structures import ImageSize
from pedrec.networks.net_pedrec.pedrec_net_temporal import PedRecNetTemporal
from pedrec.networks.net_pedrec.pedrec_net_temporal3d import PedRecNetTemporal3D
from pedrec.utils.torch_utils.torch_helper import get_device


# def net_to_mtl():
#     # init with Imagenet
#     pose_cfg = PedRecNet50Config()
#     mtl_net = PedRecLossModule(PedRecNet(pose_cfg), None)
#     weights = torch.load("data/models/pedrec_net_v3.pth")
#     # net.load_state_dict(torch.load("data/models/pedrec_net.pth"))
#
#     # freeze_layers(net.feature_extractor.children())
#     # net.pose_head.pose_confs.init_weights()
#     net_state_dict = mtl_net.state_dict()
#     for (name, param), (p_name, p_param) in zip(mtl_net.state_dict().items(), weights.items()):
#         net_state_dict[name].copy_(p_param)
#     torch.save(net_state_dict, "data/models/pedrec_net_v3_mtl.pth")

def merge_pedrec_temporal():
    # init with Imagenet
    net = PedRecNetTemporal3D(PedRecNet50Config(), ImageSize(1920, 1080))
    pedrec_weights = torch.load("data/models/pedrec/v5_net.pth")
    temporal_weights = torch.load("data/models/pedrec/temporal/pedrec_temporal_27_3d_in_3d_out.pth")
    # net.load_state_dict(torch.load("data/models/pedrec_net.pth"))

    # freeze_layers(net.feature_extractor.children())
    # net.pose_head.pose_confs.init_weights()
    net_state_dict = net.state_dict()
    for idx, (name, param) in enumerate(net.state_dict().items()):
        if name in pedrec_weights:
            net_state_dict[name].copy_(pedrec_weights[name])
        else:
            temporal_name = name.replace("head_pose_temporal.", "")
            if temporal_name in temporal_weights:
                net_state_dict[name].copy_(temporal_weights[temporal_name])
            else:
                a = 1
    torch.save(net_state_dict, "data/models/pedrec/pedrec_temporal_3d.pth")


# def load_weights_based_on_layer_order():
#     # init with Imagenet
#     net = PedRecNet(PedRecNet50Config())
#     weights = torch.load("data/models/pedrec_net_v3_net.pth")
#     # net.load_state_dict(torch.load("data/models/pedrec_net.pth"))
#
#     # freeze_layers(net.feature_extractor.children())
#     net.pose_head.pose_confs.init_weights()
#     net_state_dict = net.state_dict()
#     for (name, param), (p_name, p_param) in zip(net.state_dict().items(), weights.items()):
#         net_state_dict[name].copy_(p_param)
#     torch.save(net_state_dict, "data/models/pedrec_net_v4.pth")

# def mtlv2_to_mtlv3():
#     pose_cfg = PedRecNet50Config()
#     pose_weights = "data/models/pedrec_net_v2_mtl.pth"
#
#     mtl_net = PedRecLossModule(PedRecNet(pose_cfg), None)
#     mtl_net.load_state_dict(torch.load(pose_weights))
#
#     net = PedRecLossModule(PedRecNet(PedRecNet50Config()), None)
#     net.model.meta_head.init_weights()
#     net.model.orientation_head.init_weights()
#     net_state_dict = net.state_dict()
#     for name, param in mtl_net.state_dict().items():
#         if name == "loss_head.sigmas":
#             sigmas = net_state_dict[name]
#             sigmas[:2] = param
#             sigmas[2:4] = param # just copy the values for init ..
#         else:
#             net_state_dict[name].copy_(param)
#     torch.save(net_state_dict, "data/models/pedrec_net_v3_mtl.pth")


# def load_skip_new_stuff():
#     # init with Imagenet
#     net = PedRecNet(PedRecNet50Config())
#     weights = torch.load("data/models/pedrec/v3.0/pedrec_net_v3.pth")
#     # net.load_state_dict(torch.load("data/models/pedrec_net.pth"))
#
#     # freeze_layers(net.feature_extractor.children())
#     net.pose_head.pose_confs.init_weights()
#     net.orientation_head.init_weights()
#     net_state_dict = net.state_dict()
#     for name, param in net.state_dict().items():
#         if name in weights:
#             net_state_dict[name] = weights[name]
#         else:
#             print(f"Skipped {name}")
#     # for (name, param), (p_name, p_param) in zip(net.state_dict().items(), weights.items()):
#     #     net_state_dict[name].copy_(p_param)
#     torch.save(net_state_dict, "data/models/pedrec/pedrec_net_v3.pth")

# def load_skip_new_stuff_mtl():
#     # init with Imagenet
#     # net = PedRecNetMtl()
#     net = PedRecLossModule(PedRecNet(PedRecNet50Config()), get_device())
#     weights = torch.load("data/models/pedrec/pedrec_net_v6_mtl.pth")
#     # net.load_state_dict(torch.load("data/models/pedrec_net_v3_mtl.pth"))
#
#     # freeze_layers(net.feature_extractor.children())
#     net.model.pose_head.pose_confs.init_weights()
#     net.model.pose_head_3d.init_weights()
#     net.model.orientation_head.init_weights()
#     net.model.meta_head.init_weights()
#     net_state_dict = net.state_dict()
#     for name, param in net.state_dict().items():
#         if "sigma" in name:
#             print(f"Skipped {name}")
#             continue
#         if name in weights and net_state_dict[name].shape == weights[name].shape:
#             net_state_dict[name] = weights[name]
#         else:
#             print(f"Skipped {name}")
#     # for (name, param), (p_name, p_param) in zip(net.state_dict().items(), weights.items()):
#     #     net_state_dict[name].copy_(p_param)
#     torch.save(net_state_dict, "data/models/pedrec/pedrec_net_v6b_mtl.pth")
#
if __name__ == "__main__":
    # load_from_MTL()
    # load_weights_based_on_layer_order()
    # mtlv2_to_mtlv3()
    # load_skip_new_stuff_mtl()
    # net_to_mtl()
    merge_pedrec_temporal()