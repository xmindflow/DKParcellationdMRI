from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from typing import Union, Tuple, List
from torch import nn
from monai.networks.nets import SwinUNETR
import os
import torch.distributed as dist


class SwinUnetrBratsTrainer(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            device,
        )
        self.enable_deep_supervision = False
        # torch.set_float32_matmul_precision('high')
        # self.initial_lr = args.lr
        self.best_dsc = 0
        self.configuration_manager.configuration["patch_size"] = [128, 128, 128]
        self.configuration_manager.configuration["batch_size"] = 2
        self.batch_size = 2
        print(f"batch size: {self.configuration_manager.batch_size}")
        print(f"patch size: {self.configuration_manager.patch_size}")

    @staticmethod
    def build_network_architecture(
        # self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        v2 = False  # unetr_v2 or normal
        # self.print_to_log_file(
        #     f"Building network architecture USING SwinUnetr{'V2' if v2 else ''} for BRATS Dataset"
        # )
        # self.print_to_log_file(f"patch size: {self.configuration_manager.patch_size}")
        # print(f"crop size: {arch_init_kwargs['crop_size']}")
        return SwinUNETR(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            # img_size=self.configuration_manager.patch_size,
            img_size=[128, 128, 128],  # hardcoded for brats
            spatial_dims=3,
            use_v2=v2,
            feature_size=48,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        pass
