import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class Gaze4HRITorchDataset(Dataset):
    def __init__(self, h5_file_paths, n_frames=20, transform=None):
        """
        Args:
            h5_file_paths (list of str): List of absolute paths to the subject .h5 files.
            n_frames (int): The sliding window size.
            transform (callable, optional): Optional torchvision transforms to be applied on a sample.
        """
        self.files = h5_file_paths
        self.n_frames = n_frames
        self.transform = transform
        self.samples = [] 
        
        # Build global index of valid target frames across all subjects
        for f_idx, f_path in enumerate(self.files):
            with h5py.File(f_path, 'r') as h5:
                if 'exp_boundaries' not in h5:
                    continue
                boundaries = h5['exp_boundaries'][:]
                for start, end in boundaries:
                    # Valid if it can form a full N-frame window
                    if (end - start + 1) >= self.n_frames:
                        # Target frame goes from (start + N - 1) to end
                        for target in range(start + self.n_frames - 1, end + 1):
                            self.samples.append((f_idx, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        f_idx, target_idx = self.samples[i]
        
        with h5py.File(self.files[f_idx], 'r', swmr=True) as h5:
            # Slicing the sliding window: returns [N, 224, 224, 3]
            imgs = h5['images'][target_idx - self.n_frames + 1 : target_idx + 1]
            
            # Ground truth for the final frame in the sequence
            gaze = h5['gaze_pitch_yaw'][target_idx]
            is_blink = h5['is_blink'][target_idx]
            
        # Convert to Tensor [Sequence, Channels, H, W]
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
        
        if self.transform:
            imgs = self.transform(imgs)
            
        return {
            "images": imgs, 
            "gaze_pitch_yaw": torch.tensor(gaze),
            "is_blink": torch.tensor(is_blink, dtype=torch.float32)
        }
