import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple

class OpticalFlowAnalyzer:
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
    
    def compute_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """Compute dense optical flow using Lucas-Kanade method"""
        prev = cv2.GaussianBlur(prev_frame.astype(np.float32), (5,5), 0)
        curr = cv2.GaussianBlur(curr_frame.astype(np.float32), (5,5), 0)
        
        # Calculate gradients
        Ix = cv2.Sobel(prev, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(prev, cv2.CV_32F, 0, 1, ksize=3)
        It = curr - prev
        
        flow = np.zeros((*prev.shape, 2), dtype=np.float32)
        half_win = self.window_size // 2
        
        for y in range(half_win, prev.shape[0]-half_win):
            for x in range(half_win, prev.shape[1]-half_win):
                ix = Ix[y-half_win:y+half_win+1, x-half_win:x+half_win+1].flatten()
                iy = Iy[y-half_win:y+half_win+1, x-half_win:x+half_win+1].flatten()
                it = It[y-half_win:y+half_win+1, x-half_win:x+half_win+1].flatten()
                
                A = np.vstack((ix, iy)).T
                b = -it
                
                try:
                    flow[y,x] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    flow[y,x] = [0, 0]
        
        return flow

def visualize_optical_flow_results(prev_frame: np.ndarray,
                                 curr_frame: np.ndarray,
                                 flow: np.ndarray,
                                 frame_num: int = 0,
                                 output_dir: str = "output"):
    """Visualize optical flow results comprehensively"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a large figure with subplots
    plt.figure(figsize=(18, 12))
    
    # Plot original frames
    plt.subplot(2, 3, 1)
    plt.imshow(prev_frame, cmap='gray')
    plt.title(f'Previous Frame {frame_num}')
    
    plt.subplot(2, 3, 2)
    plt.imshow(curr_frame, cmap='gray')
    plt.title(f'Current Frame {frame_num+1}')
    
    # Plot optical flow vectors
    plt.subplot(2, 3, 3)
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h:10, 0:w:10]  # Sample every 10th pixel
    u = flow[::10, ::10, 0] * 5  # Scale for visibility
    v = flow[::10, ::10, 1] * 5
    
    plt.quiver(x, y, u, v, color='red', angles='xy', scale_units='xy', scale=1)
    plt.gca().invert_yaxis()
    plt.title('Optical Flow Vectors')
    plt.grid(True, alpha=0.3)
    
    # Plot flow components
    plt.subplot(2, 3, 4)
    plt.imshow(flow[...,0], cmap='jet')
    plt.colorbar()
    plt.title('Horizontal Flow Component')
    
    plt.subplot(2, 3, 5)
    plt.imshow(flow[...,1], cmap='jet')
    plt.colorbar()
    plt.title('Vertical Flow Component')
    
    # Plot magnitude
    plt.subplot(2, 3, 6)
    magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    plt.imshow(magnitude, cmap='hot')
    plt.colorbar()
    plt.title('Flow Magnitude')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/optical_flow_results_{frame_num:02d}.png')
    plt.close()
    
    # Save individual components
    cv2.imwrite(f'{output_dir}/frame_{frame_num:02d}.png', prev_frame)
    cv2.imwrite(f'{output_dir}/frame_{frame_num+1:02d}.png', curr_frame)
    np.save(f'{output_dir}/flow_{frame_num:02d}.npy', flow)

def process_selected_frames(video_path: str, output_dir: str = "output"):
    """Process only first 3-4 frames of video"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    analyzer = OpticalFlowAnalyzer()
    frame_count = 0
    
    # Process only 3-4 frames
    while frame_count < 3:
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow
        flow = analyzer.compute_flow(prev_gray, curr_gray)
        
        # Visualize comprehensive results
        visualize_optical_flow_results(prev_gray, curr_gray, flow, frame_count, output_dir)
        
        prev_gray = curr_gray.copy()
        frame_count += 1
        print(f"Processed frame pair {frame_count}")
    
    cap.release()
    print(f"\nDone! Results saved in '{output_dir}' folder")
    print(f"Processed {frame_count} frame pairs")

if __name__ == "__main__":
    video_path = "SampleVideo_720x480_1mb.mp4"  
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Creating sample video...")
        # Create sample video with actual motion
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('sample_video.mp4', fourcc, 20.0, (640,480))
        for i in range(40):
            frame = np.zeros((480,640,3), dtype=np.uint8)
            cv2.circle(frame, (50+i*5, 240), 30, (255,255,255), -1)  # Moving circle
            out.write(frame)
        out.release()
        video_path = "sample_video.mp4"
        print("Created sample_video.mp4 with moving object")
    
    process_selected_frames(video_path)