import os
from glob import glob
import shutil

join = os.path.join

current_path = os.getcwd()
nnunet_path = os.path.join(current_path, "nnUNet")
source_path = os.path.join(current_path, "src")

########### Copying files to nnUNet directory ###########
# Copying Training files
trainer_files = glob(os.path.join(source_path, "*Trainer.py"))
for file in trainer_files:
    shutil.copy(
        file,
        join(
            nnunet_path, "nnunetv2", "training", "nnUNetTrainer", os.path.basename(file)
        ),
    )

######### Copying Model files ###########
model_folder = os.path.join(source_path, "mednextv1")
### copy the folder to nnUNet model_sharing
model_destination = os.path.join(nnunet_path, "nnunetv2", "model_sharing", "mednextv1")
if not os.path.exists(model_destination):
    os.makedirs(model_destination)
shutil.copytree(model_folder, model_destination, dirs_exist_ok=True)

print("Files copied successfully to nnUNet directory.")
