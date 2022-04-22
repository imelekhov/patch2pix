import os
from os import path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.datasets.utils import HomographyAugmenter


class PhototourismHmgrphyDataset(Dataset):
    def __init__(self, img_path, crop_size=256, split_type="train", transforms=None):
        self.data_path = img_path
        self.split_type = split_type

        self.train_scenes = [
            "brandenburg_gate",
            "buckingham_palace",
            "colosseum_exterior",
            "grand_place_brussels",
            "notre_dame_front_facade",
            "palace_of_westminster",
            "pantheon_exterior",
            "taj_mahal",
            "temple_nara_japan",
            "trevi_fountain",
        ]

        self.val_scenes = ["sacre_coeur", "st_peters_square", "reichstag"]

        self.h_crop, self.w_crop = (crop_size, crop_size)

        self.transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Synthetic homography class
        self.homography = HomographyAugmenter(crop_hw=(self.h_crop, self.w_crop))

        self.fnames = (
            self._create_train_split()
            if self.split_type == "train"
            else self._create_val_split()
        )

    def _create_val_split(self):
        fnames = []

        for scene_name in self.val_scenes:
            for (dirpath, dirnames, filenames) in os.walk(
                self.data_path / "imw2020-valid" / scene_name / "set_100" / "images"
            ):
                for fname in filenames:
                    if fname.endswith(".jpg") and fname[0] != ".":
                        fnames.append(osp.join(dirpath, fname))
        return fnames

    def _create_train_split(self):
        fnames = []

        for scene_name in self.train_scenes:
            for (dirpath, dirnames, filenames) in os.walk(
                self.data_path / "imw2020-train" / scene_name / "dense" / "images"
            ):
                for fname in filenames:
                    if fname.endswith(".jpg") and fname[0] != ".":
                        fnames.append(osp.join(dirpath, fname))
        return fnames

    def _generate_crops(self, img):
        w, h = img.size
        cv_img2crop1, _, h2img, _, crop_center = self.homography.get_random_homography(
            image_hw=(h, w)
        )
        cv_img2crop2, _, _, h2crop2, _ = self.homography.get_random_homography(
            image_hw=(h, w), crop_center=crop_center
        )
        crop1 = self.homography.warp_image(img, cv_img2crop1)
        crop2 = self.homography.warp_image(img, cv_img2crop2)

        a_hmg_b = torch.from_numpy(np.matmul(h2img, h2crop2)).float()
        b_hmg_a = torch.inverse(a_hmg_b)
        return {
            "crop_src": crop1,
            "crop_trg": crop2,
            "a_hmg_b": a_hmg_b,
            "b_hmg_a": b_hmg_a,
        }

    def _is_mask_valid(self, grid_norm):
        # the mask
        mask = (
            (grid_norm[0, :, :, 0] >= -1.0)
            & (grid_norm[0, :, :, 0] <= 1.0)
            & (grid_norm[0, :, :, 1] >= -1.0)
            & (grid_norm[0, :, :, 1] <= 1.0)
        )
        valid = not ((mask == 0).sum() == self.h_crop * self.w_crop)
        return valid

    def __getitem__(self, item):
        img_fname = self.fnames[item]
        img = Image.open(img_fname).convert("RGB")
        crops_data = self._generate_crops(img)

        crop1 = crops_data["crop_src"]
        crop2 = crops_data["crop_trg"]

        if self.transforms:
            crop1 = self.transforms(image=crop1)["image"]
            crop2 = self.transforms(image=crop2)["image"]

        return {
            "img_fname": self.fnames[item],
            "crop_src": crop1,
            "crop_trg": crop2,
            "h_mtx": crops_data["b_hmg_a"],
        }

    def __len__(self):
        return len(self.fnames)
