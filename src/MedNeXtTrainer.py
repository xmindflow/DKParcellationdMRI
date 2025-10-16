from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from typing import Union, Tuple, List
from torch import nn
from nnunetv2.model_sharing.mednextv1.create_mednext_v1 import create_mednextv1_medium
import os
import torch.distributed as dist
from torch._dynamo import OptimizedModule
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


class MednextBratsTrainer(nnUNetTrainer):
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
        self.initial_lr = 0.001
        self.best_dsc = 0
        self.configuration_manager.configuration["patch_size"] = [128, 128, 128]
        self.configuration_manager.configuration["batch_size"] = 2
        self.batch_size = 2
        self.configuration_manager.configuration["pool_op_kernel_sizes"] = [
            [4, 4, 4],
            [2, 2, 2],
            [2, 2, 2],
        ]
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
        # self.print_to_log_file(
        #     f"Building network architecture USING MedNeXt for BRATS Dataset"
        # )
        # self.print_to_log_file(f"patch size: {self.configuration_manager.patch_size}")
        # print(f"crop size: {arch_init_kwargs['crop_size']}")
        return create_mednextv1_medium(
            num_input_channels=num_input_channels,
            num_classes=num_output_channels,
            ds=enable_deep_supervision,
            kernel_size=3,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        pass
        # if self.is_ddp:
        #     mod = self.network.module
        # else:
        #     mod = self.network
        # if isinstance(mod, OptimizedModule):
        #     mod = mod._orig_mod

        # mod.do_ds = enabled

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-4,  # 1e-8 might cause nans in fp16
        )

        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
