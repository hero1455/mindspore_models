"""preprocess"""
import os

from src.dataset import create_dataset
from model_utils.config import config

def get_bin():
    """generate bin files"""
    ds_eval, _, _ = create_dataset(test_train=False, data_dir=config.data_path,
                                   dataset=config.dataset, train_epochs=0,
                                   eval_batch_size=config.eval_batch_size)
    bs = config.eval_batch_size
    user_folder = os.path.join(config.pre_result_path, "00_user")
    os.makedirs(user_folder)
    item_folder = os.path.join(config.pre_result_path, "01_item")
    os.makedirs(item_folder)
    mask_folder = os.path.join(config.pre_result_path, "02_mask")
    os.makedirs(mask_folder)

    for i, dataset in enumerate(ds_eval.create_tuple_iterator(output_numpy=True)):
        users, items, masks = dataset
        file_name = "ncf_bs" + str(bs) + "_" + str(i) + ".bin"
        users_path = os.path.join(user_folder, file_name)
        users.tofile(users_path)
        items_path = os.path.join(item_folder, file_name)
        items.tofile(items_path)
        masks_path = os.path.join(mask_folder, file_name)
        masks.tofile(masks_path)
    print("=" * 20, "Export bin files success", "=" * 20)

if __name__ == '__main__':
    get_bin()
