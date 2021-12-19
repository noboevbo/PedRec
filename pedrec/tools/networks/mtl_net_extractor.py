import sys
sys.path.append(".")
import torch

from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.utils.torch_utils.torch_helper import get_device
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet, PedRecNetLossHead
from pedrec.networks.net_pedrec.pedrec_net_mtl_wrapper import PedRecNetMTLWrapper


def mtl_to_net(mtl_wrapper: torch.nn.Module, mtl_path: str, output_path: str):
    mtl_wrapper.load_state_dict(torch.load(mtl_path))
    torch.save(mtl_wrapper.model.state_dict(), output_path)

if __name__ == "__main__":
    pose_cfg = PedRecNet50Config()
    device = get_device(True)
    net = PedRecNetMTLWrapper(PedRecNet(pose_cfg), PedRecNetLossHead(device))
    # mtl_weights = "data/models/pedrec/single_results/experiment_pedrec_direct_finetune_tud_with_tud_0.pth"
    # output_path = "data/models/pedrec/single_results/experiment_pedrec_direct_finetune_tud_with_tud_0_net.pth"
    # mtl_weights = "data/models/pedrec/experiment_pedrec_p2d3d_c_h36m_sim_0.pth"
    # output_path = "data/models/pedrec/experiment_pedrec_p2d3d_c_h36m_sim_0_net.pth"
    # mtl_weights = "data/models/pedrec/experiment_pedrec_direct_4.pth"
    # output_path = "data/models/pedrec/experiment_pedrec_direct_4_net.pth"
    # mtl_weights = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_tud_0.pth"
    # output_path = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_tud_0_net.pth"
    # mtl_weights = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_h36m_tud_0.pth"
    # output_path = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_h36m_tud_0_net.pth"

    mtl_weights = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_sim_0.pth"
    output_path = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_sim_0_net.pth"


    mtl_to_net(net, mtl_weights, output_path)
