import socket
import msgpack
import msgpack_numpy as m
import numpy as np
from gaze_estimation import GazeModel
from tqdm import tqdm
import cv2

m.patch()

def visualize_clip(frames, window_name="MCGaze Clip", scale=0.5, wait=500):
    """
    Visualize a list of frames side by side in one window, then close it. for debugging. you can use in estimate_from_crops as:
    if idx % self.clip_size == 0:
        visualize_clip(clip, window_name="Debug Clip", scale=0.4, wait=200)

    Args:
        frames (list[np.ndarray]): List of frames (H,W,3), BGR.
        window_name (str): Name of the window.
        scale (float): Resize factor for visualization.
        wait (int): Wait time in ms before closing (e.g. 500 ms).
    """
    if len(frames) == 0:
        print("⚠️ No frames to visualize")
        return

    resized = []
    for f in frames:
        if scale != 1.0:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        resized.append(f)

    concat = np.concatenate(resized, axis=1)

    cv2.imshow(window_name, concat)
    cv2.waitKey(wait)  # show for 'wait' ms
    cv2.destroyWindow(window_name)  # close this window

# --- in gaze_model_mcgaze.py ---

class MCGaze(GazeModel):
    def __init__(self, clip_size: int, rectification: bool, socket_path: str = "/tmp/mcgaze_server.sock"):
        super().__init__(rectification=rectification)
        self.clip_size = clip_size
        self.rectification = rectification
        self.socket_path = socket_path
        self._full_gaze_tensor = None  # will hold [N, clip_size, 3] for saving

    def get_model_name(self) -> str:
        return "mcgaze_clip_size_" + str(self.clip_size) + ("_rectification" if self.rectification else "")

    def estimate_from_crops(self, head_crops, timestamps):
        """
        Returns:
            gaze_directions: [N, 4]  -> [timestamp, gx, gy, gz] for the *current* frame at each step
            valid_indices:   [M]     -> indices where a new prediction was produced
        Side effect:
            self._full_gaze_tensor: [N, clip_size, 3] right-aligned per step, left zero-padded
        """
        N = len(head_crops)
        M = int(self.clip_size)

        filled = []                                  # Nx4 rolling output (timestamp + last-frame gaze)
        valid_idx = []                               # indices with fresh last-frame prediction
        full_tensor = np.zeros((N, M, 3), dtype=np.float64)  # right-align per step, left zero-pad

        last_valid_gaze = None

        for idx, (crop, ts) in enumerate(tqdm(zip(head_crops, timestamps), total=N, desc="MCGaze Estimation")):
            # Build contiguous rolling window up to M (stride=1)
            start = max(0, idx - (M - 1))
            clip = head_crops[start:idx + 1]
            if len(clip) > 0 and False:  # for debugging use False instead of  True
                self._save_model_input_frame(clip[-1], idx)
            # Send exactly the frames you have; do not pad/duplicate
            result = self.send_head_crop_batch(clip, clip_size=len(clip))

            # Right-align the server outputs into a fixed-length row [M,3] with left zeros
            k = len(result) if isinstance(result, list) else 0
            if k > 0:
                for j, r in enumerate(result):
                    vec = r.get("gaze_vector")
                    if vec is None:
                        vec = [0.0, 0.0, 0.0]  # safety fallback
                    full_tensor[idx, M - k + j, :] = np.asarray(vec, dtype=np.float64)

                # Stream output uses the last frame's vector
                latest_vec = result[-1].get("gaze_vector")
                if latest_vec is not None:
                    last_valid_gaze = latest_vec
                    valid_idx.append(idx)
                    filled.append([float(ts)] + list(latest_vec))
                elif last_valid_gaze is not None:
                    filled.append([float(ts)] + list(last_valid_gaze))
            else:
                # IPC hiccup: keep zeros in full_tensor row; carry forward last valid for stream
                if last_valid_gaze is not None:
                    filled.append([float(ts)] + list(last_valid_gaze))

        # Stash full tensor for save_estimation()
        self._full_gaze_tensor = full_tensor

        return np.array(filled, dtype=np.float64), np.array(valid_idx, dtype=np.int32)

    def save_estimation(self, gaze_directions: np.ndarray, valid_indices: np.ndarray,
                        output_dir: str, append_model_name_to_output_path=True):
        """
        Keeps the original saves (gaze_directions.npy, gaze_directions_indices.npy)
        and additionally writes gaze_directions_full.npy if available via self._full_gaze_tensor.
        """
        import os
        # call base to save the standard files
        super().save_estimation(gaze_directions, valid_indices, output_dir, append_model_name_to_output_path)

        base_output_path = output_dir
        if append_model_name_to_output_path:
            base_output_path = os.path.join(base_output_path, self.get_model_name())
        os.makedirs(base_output_path, exist_ok=True)

        if getattr(self, "_full_gaze_tensor", None) is not None:
            full_path = os.path.join(base_output_path, "gaze_directions_full.npy")
            np.save(full_path, self._full_gaze_tensor)


    def send_msg(self, sock, obj):
        payload = msgpack.packb(obj, use_bin_type=True)
        sock.sendall(len(payload).to_bytes(4, 'big'))
        sock.sendall(payload)

    def recv_msg(self, sock):
        msg_len = int.from_bytes(sock.recv(4), 'big')
        data = b''
        while len(data) < msg_len:
            packet = sock.recv(msg_len - len(data))
            if not packet:
                break
            data += packet
        return msgpack.unpackb(data, raw=False)

    def send_head_crop_batch(self, head_crop_batch, clip_size):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(self.socket_path)
            msg = {
                "clip_size": clip_size,
                "head_crops": head_crop_batch
            }
            self.send_msg(client, msg)
            result = self.recv_msg(client)
            return result
        except Exception as e:
            print("Client error:", e)
            return []
        finally:
            client.close()

    def _save_model_input_frame(self, frame_rgb: np.ndarray, idx: int,
                                save_dir: str = "/home/kovan/USTA/mcgaze_debug_frames", max_images: int = 12):
        """
        Save the model input frame (last frame of the clip) as a PNG.
        Assumes RGB input; converts to BGR for cv2.imwrite.
        """
        if not hasattr(self, "_viz_saved"):
            self._viz_saved = 0
        if self._viz_saved >= max_images:
            return

        if frame_rgb is None or frame_rgb.size == 0:
            return

        import os
        os.makedirs(save_dir, exist_ok=True)

        fr = frame_rgb
        if fr.dtype != np.uint8:
            fr = np.clip(fr, 0, 255).astype(np.uint8)

        # RGB -> BGR for OpenCV
        bgr = fr[..., ::-1]
        out_path = os.path.join(save_dir, f"model_input_{self._viz_saved:04d}_i{idx}.png")
        cv2.imwrite(out_path, bgr)
        self._viz_saved += 1
