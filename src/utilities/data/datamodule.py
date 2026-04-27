import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
from src.latent_diffusion.util import instantiate_from_config
from utilities.data.dataset import AudiostockDataset, DS_10283_2325_Dataset, Audiostock_splited_Dataset, Slakh_Dataset, MultiSource_Slakh_Dataset, MultiSource_Slakh_Waveform_Dataset, MultiSource_Slakh_Inference_Dataset, ADTOFDataset, ADTOFInferenceDataset, MDBDrumsSingleFileDataset, StemGMDOnsetDataset
from utilities.data.dataset import ENSTOnsetDataset
import torch
import omegaconf




class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size=2, num_workers=1, augmentation= None, path=None, preprocessing=None, config = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers # if num_workers is not None else batch_size*2
        self.path = path
        self.augmentation=augmentation
        self.preprocessing=preprocessing
        
        self.config = {}
        self.config["path"] = path
        self.config["preprocessing"] = preprocessing
        self.config["augmentation"] = augmentation

        # Safely check for 'shuffle_val_test' in the config and set it to False if not found
        self.shuffle_val_test = self.config.get("path", {}).get("shuffle_val_test", False)

        self.pin_memory = False
        self.persistent_workers = False
        self.prefetch_factor = 2
        self.loader_timeout = 0


    def prepare_data(self):
        # This method is used for data download and preprocessing (if any)
        # It is called only once across all GPUs in a distributed setup
        # You can use this method to download the data or perform any necessary preprocessing steps.
        pass       

    def setup(self, stage=None):

        if 'train_data' in self.path and self.path['train_data'] is not None:
            self.train_dataset = self.load_dataset(self.path['train_data'], split = "train")
            if self.train_dataset is not None:
                if hasattr(self.path, 'split') and self.path['split']:
                    if self.path['split'] == "test":
                        self.train_dataset, self.test_dataset = self.split_data(self.train_dataset) #sometime we want opur split to be test for some technical reasons
                    else:
                        self.train_dataset, self.val_dataset = self.split_data(self.train_dataset)
        
        if 'valid_data' in self.path and self.path['valid_data'] is not None:
            self.val_dataset  = self.load_dataset(self.path['valid_data'], split = "valid") 
        
        if 'test_data' in self.path and self.path['test_data'] is not None:
            self.test_dataset = self.load_dataset(self.path['test_data'], split = "test") 

        has_valid_dataset = (hasattr(self, 'train_dataset') and self.train_dataset is not None) or \
                           (hasattr(self, 'val_dataset') and self.val_dataset is not None) or \
                           (hasattr(self, 'test_dataset') and self.test_dataset is not None)
        
        if not has_valid_dataset:
            raise ValueError("Invalid dataset configuration provided. No valid datasets found.")

    def _get_dataset_cls_by_type(self, dtype: str):
        if dtype == "Audiostock":
            return AudiostockDataset
        if dtype == "DS_10283_2325":
            return DS_10283_2325_Dataset
        if dtype == "Audiostock_splited":
            return Audiostock_splited_Dataset
        if dtype == "Slakh":
            return Slakh_Dataset
        if dtype == "MultiSource_Slakh":
            return MultiSource_Slakh_Dataset
        if dtype == "MultiSource_Slakh_Waveform":
            return MultiSource_Slakh_Waveform_Dataset
        if dtype == "MultiSource_Slakh_Inference":
            return MultiSource_Slakh_Inference_Dataset
        if dtype == "ADTOF":
            return ADTOFDataset
        if dtype == "ADTOFInference":
            return ADTOFInferenceDataset
        if dtype == "MDBDrumsSingleFile":
            return MDBDrumsSingleFileDataset
        if dtype == "StemGMDOnset":
            return StemGMDOnsetDataset
        if dtype == "ENSTOnset":
            return ENSTOnsetDataset
        raise ValueError(f"Unsupported dataset_type: {dtype}")


    def load_dataset(self, path, split = "train"):
        from torch.utils.data import ConcatDataset

        whole_track = self.config.get("path", {}).get("whole_track", False)
        print(f"[DataModule] {split} dataset whole_track setting: {whole_track}")

        def _build_one(ds_path, ds_type, ds_label_path, is_train):
            if not ds_path:
                return None
            ds_cls = self._get_dataset_cls_by_type(ds_type)
            print(f"[DataModule] preparing to load {ds_type}: path={ds_path}, label={ds_label_path}")
            return ds_cls(
                dataset_path=ds_path,
                label_path=ds_label_path,
                config=self.config,
                train=is_train,
                factor=1.0,
                whole_track=whole_track,
            )

        if not isinstance(path, (list, omegaconf.listconfig.ListConfig)):
            if not path:
                print(f"[DataModule] {split} dataset is empty, skip loading")
                return None
            dtype = self.path["dataset_type"]
            label_path = self.config["path"].get("label_data", None)
            is_train = (split == "train")
            return _build_one(path, dtype, label_path, is_train)

        paths = list(path)
        dtypes = self.path["dataset_type"]
        if not isinstance(dtypes, (list, omegaconf.listconfig.ListConfig)):
            dtypes = [dtypes] * len(paths)
        else:
            dtypes = list(dtypes)

        label_cfg = self.config["path"].get("label_data", None)
        if isinstance(label_cfg, (list, omegaconf.listconfig.ListConfig)):
            label_list = list(label_cfg)
        else:
            label_list = [label_cfg] * len(paths)

        datasets = []
        for i, pth in enumerate(paths):
            ds = _build_one(pth, dtypes[i], label_list[i] if i < len(label_list) else None, is_train=(split == "train"))
            if ds is not None:
                datasets.append(ds)

        if len(datasets) == 0:
            print(f"[DataModule] {split} no valid dataset can be loaded")
            return None
        if len(datasets) == 1:
            return datasets[0] 
        print(f"[DataModule] {split} mixed {len(datasets)} datasets (ConcatDataset)")
        return ConcatDataset(datasets)

    def split_data(self, dataset):
        # Split the dataset into train, validation, and test sets
        train_len = int(0.9 * len(dataset))
        val_len = int(0.1 * len(dataset))

        train_dataset, val_dataset = torch.utils.data.random_split(
                                    dataset, [train_len, val_len], 
                                    generator=torch.Generator().manual_seed(42))
        print(f"Dataset has been splitted! Train: {len(train_dataset)}, Valid: {len(val_dataset)}")

        return train_dataset, val_dataset

    def train_dataloader(self):
        # Returns the DataLoader for the training dataset
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            timeout=self.loader_timeout
        )

    def val_dataloader(self):
        # Returns the DataLoader for the validation dataset
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=self.shuffle_val_test,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            timeout=self.loader_timeout
        )

    def test_dataloader(self):
        # Returns the DataLoader for the test dataset
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=self.shuffle_val_test,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            timeout=self.loader_timeout
        )