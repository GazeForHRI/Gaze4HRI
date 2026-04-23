import os
import pickle
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from exordium.utils.normalize import standardization
from blinklinmult.models import BlinkLinMulT

class Gaze4HRISequenceDataset(Dataset):
    def __init__(self, pkl_path, json_path, seq_len=15, stride=1):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Exordium landmarks not found at {pkl_path}. Please run the preprocessing pipeline first.")
            
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        
        with open(json_path, 'r') as f:
            self.mean_std = json.load(f)
            
        self.seq_len = seq_len
        self.stride = stride
        self.num_frames = len(self.data)
        
        # Calculate how many full sequences of length seq_len we can extract
        self.valid_windows = max(0, (self.num_frames - self.seq_len) // self.stride + 1)
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.valid_windows

    def _get_eye_features(self, sample_window, eye_prefix):
        low_level = torch.stack([
            self.transform(Image.fromarray(elem[f'annotation_{eye_prefix}_eye_features']['eye'])) 
            for elem in sample_window
        ], dim=0)
        
        headpose = np.array([elem['tddfa-retinaface_headpose'] for elem in sample_window])
        landmarks = np.array([elem[f'annotation_{eye_prefix}_eye_features']['landmarks'] for elem in sample_window]).reshape(self.seq_len, -1)
        iris_landmarks = np.array([elem[f'annotation_{eye_prefix}_eye_features']['iris_landmarks'] for elem in sample_window]).reshape(self.seq_len, -1)
        iris_diameters = np.array([elem[f'annotation_{eye_prefix}_eye_features']['iris_diameters'] for elem in sample_window])
        eyelid_pupil = np.array([elem[f'annotation_{eye_prefix}_eye_features']['eyelid_pupil_distances'] for elem in sample_window])
        ear = np.expand_dims(np.array([elem[f'annotation_{eye_prefix}_eye_features']['ear'] for elem in sample_window]), axis=-1)

        # --- EXTENSIVE DEBUGGING BLOCK ---
        if getattr(self, '_debug_printed', 0) < 2:
            print(f"\n--- DEBUG INFO ({eye_prefix.upper()} EYE) ---")
            print(f"headpose: shape={headpose.shape}, dtype={headpose.dtype}")
            print(f"landmarks: shape={landmarks.shape}, dtype={landmarks.dtype}")
            print(f"iris_landmarks: shape={iris_landmarks.shape}, dtype={iris_landmarks.dtype}")
            print(f"iris_diameters: shape={iris_diameters.shape}, dtype={iris_diameters.dtype}")
            print(f"eyelid_pupil: shape={eyelid_pupil.shape}, dtype={eyelid_pupil.dtype}")
            print(f"ear: shape={ear.shape}, dtype={ear.dtype}")
            
            # Print JSON stats type for the first feature to verify
            print(f"json_headpose_mean: type={type(self.mean_std['headpose']['mean'][0])}")
            self._debug_printed = getattr(self, '_debug_printed', 0) + 1

        # --- EXPLICIT FLOAT32 CASTING ---
        high_level = torch.concat([
            standardization(torch.tensor(headpose, dtype=torch.float32), torch.tensor(self.mean_std['headpose']['mean'], dtype=torch.float32), torch.tensor(self.mean_std['headpose']['std'], dtype=torch.float32)),
            standardization(torch.tensor(landmarks, dtype=torch.float32), torch.tensor(self.mean_std['eye_landmarks']['mean'], dtype=torch.float32), torch.tensor(self.mean_std['eye_landmarks']['std'], dtype=torch.float32)),
            standardization(torch.tensor(iris_landmarks, dtype=torch.float32), torch.tensor(self.mean_std['iris_landmarks']['mean'], dtype=torch.float32), torch.tensor(self.mean_std['iris_landmarks']['std'], dtype=torch.float32)),
            standardization(torch.tensor(iris_diameters, dtype=torch.float32), torch.tensor(self.mean_std['iris_diameters']['mean'], dtype=torch.float32), torch.tensor(self.mean_std['iris_diameters']['std'], dtype=torch.float32)),
            standardization(torch.tensor(eyelid_pupil, dtype=torch.float32), torch.tensor(self.mean_std['eyelid_pupil_distances']['mean'], dtype=torch.float32), torch.tensor(self.mean_std['eyelid_pupil_distances']['std'], dtype=torch.float32)),
            standardization(torch.tensor(ear, dtype=torch.float32), torch.tensor(self.mean_std['ear']['mean'], dtype=torch.float32), torch.tensor(self.mean_std['ear']['std'], dtype=torch.float32)),
        ], dim=-1)
        
        return low_level.float(), high_level.float()

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        sample_window = [self.data[i] for i in range(start_idx, start_idx + self.seq_len)]
        
        left_low, left_high = self._get_eye_features(sample_window, 'left')
        right_low, right_high = self._get_eye_features(sample_window, 'right')
        
        return {
            'start_idx': start_idx,
            'left_low': left_low, 'left_high': left_high,
            'right_low': right_low, 'right_high': right_high
        }

def run_inference(exp_dir, json_path, seq_len=15, stride=1, weights='blinklinmult-union', batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    pkl_path = os.path.join(exp_dir, "exordium_landmarks", "exordium_data.pkl")
    dataset = Gaze4HRISequenceDataset(pkl_path, json_path, seq_len=seq_len, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = BlinkLinMulT(weights=weights).to(device)
    model.eval()
    
    all_start_idxs = []
    all_probs = []
    
    print(f"Starting distinct window inference (Seq Len: {seq_len}, Stride: {stride})...")
    with torch.no_grad():
        for batch in dataloader:
            start_idxs = batch['start_idx'].numpy()
            
            l_low, l_high = batch['left_low'].to(device), batch['left_high'].to(device)
            r_low, r_high = batch['right_low'].to(device), batch['right_high'].to(device)
            
            # Forward passes
            l_cls_logits, l_seq_logits = model([l_low, l_high])
            r_cls_logits, r_seq_logits = model([r_low, r_high])
            
            # Average the logits as per the paper methodology
            avg_seq_logits = (l_seq_logits + r_seq_logits) / 2.0
            probs = torch.sigmoid(avg_seq_logits).cpu().numpy().squeeze(-1) 
            
            all_start_idxs.extend(start_idxs)
            all_probs.append(probs)

    final_start_idxs = np.array(all_start_idxs)
    final_probs_matrix = np.concatenate(all_probs, axis=0)
    
    output_path = os.path.join(exp_dir, "exordium_landmarks", "blink_probabilities_raw.npz")
    np.savez(output_path, start_indices=final_start_idxs, probabilities=final_probs_matrix)
    print(f"Saved distinct blink probabilities to {output_path}")
    print(f"Array shapes - Start Indices: {final_start_idxs.shape}, Probabilities: {final_probs_matrix.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to the experiment directory")
    parser.add_argument("--json", type=str, required=True, help="Path to talkingface.json")
    parser.add_argument("--seq_len", type=int, default=15, help="Number of frames per sequence window")
    parser.add_argument("--stride", type=int, default=1, help="Stride between consecutive sequence windows")
    args = parser.parse_args()
    
    run_inference(args.exp_dir, args.json, args.seq_len, args.stride)
