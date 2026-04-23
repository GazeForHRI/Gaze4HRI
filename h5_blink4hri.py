import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torchvision import transforms

class Blink4HRITorchDataset(Dataset):
    def __init__(self, h5_file_paths, n_frames=15, stride=15, mean_std_dict=None):
        self.files = h5_file_paths
        self.n_frames = n_frames
        self.stride = stride
        self.mean_std_dict = mean_std_dict
        self.samples = [] 
        self.labels = [] 
        
        # Dictionary to hold open HDF5 file handles per worker
        self.h5_handles = {}
        
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for f_idx, f_path in enumerate(self.files):
            # We temporarily open the files here just to map the boundaries.
            # This happens in the main process before workers are spawned.
            with h5py.File(f_path, 'r') as h5:
                if 'exp_boundaries' not in h5:
                    continue
                    
                boundaries = h5['exp_boundaries'][:]
                is_blink_ds = h5['is_blink'][:]
                
                for start, end in boundaries:
                    if (end - start + 1) >= self.n_frames:
                        # Stride added here to prevent overlapping sequences
                        for target in range(start + self.n_frames - 1, end + 1, self.stride):
                            window_start = target - self.n_frames + 1
                            
                            self.samples.append((f_idx, target))
                            
                            sequence_label = 1 if np.any(is_blink_ds[window_start:target+1]) else 0
                            self.labels.append(sequence_label)

    def __len__(self):
        return len(self.samples)
        
    def get_labels(self):
        """Returns sequence-level labels to be used by a PyTorch WeightedRandomSampler."""
        return self.labels
        
    def _standardize_high_level(self, features):
        if self.mean_std_dict is None:
            return features
        mean = torch.tensor(self.mean_std_dict['mean'], dtype=torch.float32)
        std = torch.tensor(self.mean_std_dict['std'], dtype=torch.float32)
        return (features - mean) / (std + 1e-7)

    def _get_h5_handle(self, f_idx):
        """Lazy loading of HDF5 files per worker process."""
        if f_idx not in self.h5_handles:
            self.h5_handles[f_idx] = h5py.File(self.files[f_idx], 'r', swmr=True)
        return self.h5_handles[f_idx]

    def __getitem__(self, i):
        f_idx, target_idx = self.samples[i]
        
        # Fetch the worker-specific open file handle
        h5 = self._get_h5_handle(f_idx)
        
        start_idx = target_idx - self.n_frames + 1
        end_idx = target_idx + 1
        
        left_eye = h5['left_eye'][start_idx:end_idx]
        right_eye = h5['right_eye'][start_idx:end_idx]
        left_feats = h5['left_features'][start_idx:end_idx]
        right_feats = h5['right_features'][start_idx:end_idx]
        is_blink = h5['is_blink'][start_idx:end_idx]
            
        left_eye = torch.from_numpy(left_eye).permute(0, 3, 1, 2).float() / 255.0
        right_eye = torch.from_numpy(right_eye).permute(0, 3, 1, 2).float() / 255.0
        
        left_eye = torch.stack([self.img_transform(img) for img in left_eye])
        right_eye = torch.stack([self.img_transform(img) for img in right_eye])
        
        left_feats = torch.from_numpy(left_feats).float()
        right_feats = torch.from_numpy(right_feats).float()
        
        is_blink = torch.from_numpy(is_blink).float()

        left_feats = self._standardize_high_level(left_feats)
        right_feats = self._standardize_high_level(right_feats)

        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "left_features": left_feats,
            "right_features": right_feats,
            "is_blink": is_blink 
        }

    def __del__(self):
        """Ensure open file handles are closed when the dataset is destroyed."""
        for handle in getattr(self, 'h5_handles', {}).values():
            try:
                handle.close()
            except:
                pass
