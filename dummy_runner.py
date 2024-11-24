#all imports
import torch
import cv2
import numpy as np
from time import sleep
def dummy_runner_inference(*args, **kwargs):
    upscaled_frames = []
    frames_bw_np = torch.rand(24, 32, 32).numpy().astype(np.float32)
    print(frames_bw_np.shape)
    for i in range(args[3]):
        sleep(0.2)
        yield i
    # Upscale each frame
    for i in range(frames_bw_np.shape[0]):
        frame_np = frames_bw_np[i]  # shape [32, 32] as a numpy array
        # Resize the frame to the desired output resolution
        upscaled_frame = cv2.resize(frame_np, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        # Convert back to tensor and add the channel dimension to match [512, 512, 1]
        tmp = torch.from_numpy(upscaled_frame).unsqueeze(2).repeat(1, 1, 3)
        upscaled_frames.append(tmp)

    # Stack upscaled frames to create a single tensor of shape [24, 512, 512, 3]
    upscaled_frames_tensor = torch.stack(upscaled_frames)
    print(upscaled_frames_tensor.shape)  

    import torchvision.io as io

    # Save as video file (e.g., "output.mp4") at 8 frames per second
    output_path = f"./outputs/output_{0}_cfgs_{0}_{args[0]}.mp4"
    io.write_video(output_path, upscaled_frames_tensor, fps=8)
    print(output_path)
    
    yield output_path