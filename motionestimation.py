import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional

class MotionEstimator:
    def __init__(self, block_size: int = 16, search_range: int = 16):
        self.block_size = block_size
        self.search_range = search_range
    
    def sad(self, block1: np.ndarray, block2: np.ndarray) -> float:
        return np.sum(np.abs(block1.astype(np.float32) - block2.astype(np.float32)))
    
    def full_search_block_matching(self, 
                                 reference_frame: np.ndarray, 
                                 current_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = current_frame.shape
        bs = self.block_size
        sr = self.search_range
        
        mv_rows = h // bs
        mv_cols = w // bs
        motion_vectors = np.zeros((mv_rows, mv_cols, 2), dtype=np.int32)
        residual_frame = np.zeros_like(current_frame)
        
        for i in range(mv_rows):
            for j in range(mv_cols):
                y = i * bs
                x = j * bs
                current_block = current_frame[y:y+bs, x:x+bs]
                
                min_cost = float('inf')
                best_mv = (0, 0)
                
                for dy in range(-sr, sr + 1):
                    for dx in range(-sr, sr + 1):
                        ref_y = y + dy
                        ref_x = x + dx
                        
                        if (0 <= ref_y <= h-bs and 0 <= ref_x <= w-bs):
                            ref_block = reference_frame[ref_y:ref_y+bs, ref_x:ref_x+bs]
                            cost = self.sad(current_block, ref_block)
                            
                            if cost < min_cost:
                                min_cost = cost
                                best_mv = (dx, dy)
                
                motion_vectors[i, j] = best_mv
                dx, dy = best_mv
                ref_y = y + dy
                ref_x = x + dx
                
                if (0 <= ref_y <= h-bs and 0 <= ref_x <= w-bs):
                    residual_frame[y:y+bs, x:x+bs] = current_block - reference_frame[ref_y:ref_y+bs, ref_x:ref_x+bs]
        
        return motion_vectors, residual_frame

def visualize_results(frame1: np.ndarray, 
                    frame2: np.ndarray,
                    motion_vectors: np.ndarray,
                    residual: np.ndarray,
                    block_size: int = 16,
                    scale: float = 1.0,
                    save_path: Optional[str] = None) -> None:
    """Visualize input frames, motion vectors and residual together"""
    plt.figure(figsize=(15, 10))
    
    # Plot first frame
    plt.subplot(2, 2, 1)
    plt.imshow(frame1, cmap='gray')
    plt.title('Reference Frame')
    
    # Plot second frame
    plt.subplot(2, 2, 2)
    plt.imshow(frame2, cmap='gray')
    plt.title('Current Frame')
    
    # Plot motion vectors
    plt.subplot(2, 2, 3)
    mv_rows, mv_cols = motion_vectors.shape[:2]
    y_coords = np.arange(0, mv_rows * block_size, block_size) + block_size // 2
    x_coords = np.arange(0, mv_cols * block_size, block_size) + block_size // 2
    Y, X = np.meshgrid(x_coords, y_coords)
    U = motion_vectors[:, :, 0] * scale
    V = motion_vectors[:, :, 1] * scale
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='red', width=0.002)
    plt.gca().invert_yaxis()
    plt.title('Motion Vectors')
    plt.grid(True, alpha=0.3)
    
    # Plot residual
    plt.subplot(2, 2, 4)
    plt.imshow(residual, cmap='gray')
    plt.title('Residual Frame')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved results to {save_path}")
    else:
        try:
            plt.show()
        except:
            plt.savefig('motion_analysis_results.png')
            print("Saved results to motion_analysis_results.png")

def main():
    # Load video frames
    cap = cv2.VideoCapture('SampleVideo_720x480_1mb.mp4')
    ret, frame1 = cap.read()
    if not ret:
        print("Error loading video")
        return
    
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    ret, frame2 = cap.read()
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    cap.release()
    
    # Save original frames for visualization
    cv2.imwrite('frame1.png', frame1)
    cv2.imwrite('frame2.png', frame2)
    
    # Initialize and process
    me = MotionEstimator(block_size=16, search_range=16)
    motion_vectors, residual = me.full_search_block_matching(frame1_gray, frame2_gray)
    
    # Visualize all results together
    visualize_results(frame1_gray, frame2_gray, motion_vectors, residual,
                    save_path='motion_analysis_results.png')
    
    print("Analysis complete. Check saved images:")
    print("- frame1.png (Reference frame)")
    print("- frame2.png (Current frame)")
    print("- motion_analysis_results.png (All results together)")

if __name__ == "__main__":
    main()