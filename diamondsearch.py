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
        """Sum of Absolute Differences"""
        return np.sum(np.abs(block1.astype(np.float32) - block2.astype(np.float32)))
    
    def diamond_search(self, 
                     reference_frame: np.ndarray, 
                     current_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast motion estimation using Diamond Search
        Returns: (motion_vectors, residual_frame)
        """
        h, w = current_frame.shape
        bs = self.block_size
        
        # Initialize outputs
        mv_rows = h // bs
        mv_cols = w // bs
        motion_vectors = np.zeros((mv_rows, mv_cols, 2), dtype=np.int32)
        residual_frame = np.zeros_like(current_frame)
        
        # Diamond search patterns
        LARGE_DIAMOND = [(-2,0), (0,-2), (2,0), (0,2)]  # Step=2
        SMALL_DIAMOND = [(-1,-1), (-1,1), (1,-1), (1,1)]  # Step=1
        
        for i in range(mv_rows):
            for j in range(mv_cols):
                y = i * bs
                x = j * bs
                current_block = current_frame[y:y+bs, x:x+bs]
                
                # Start from (0,0) position
                best_dx, best_dy = 0, 0
                min_cost = self.sad(current_block, 
                                  reference_frame[y:y+bs, x:x+bs])
                
                # Diamond Search
                step_size = 2  # Start with large diamond
                while step_size >= 1:
                    pattern = LARGE_DIAMOND if step_size == 2 else SMALL_DIAMOND
                    improved = False
                    
                    for dx, dy in pattern:
                        test_x = best_dx + dx
                        test_y = best_dy + dy
                        
                        # Check search bounds
                        if (y+test_y >=0 and y+test_y+bs <= h and 
                            x+test_x >=0 and x+test_x+bs <= w):
                            
                            ref_block = reference_frame[y+test_y:y+test_y+bs, 
                                                      x+test_x:x+test_x+bs]
                            cost = self.sad(current_block, ref_block)
                            
                            if cost < min_cost:
                                min_cost = cost
                                best_dx, best_dy = test_x, test_y
                                improved = True
                    
                    # Reduce step size if no improvement
                    if not improved:
                        step_size -= 1
                
                # Store best motion vector
                motion_vectors[i,j] = [best_dx, best_dy]
                
                # Calculate residual
                ref_y = y + best_dy
                ref_x = x + best_dx
                if (0 <= ref_y <= h-bs and 0 <= ref_x <= w-bs):
                    residual_frame[y:y+bs, x:x+bs] = current_block - \
                        reference_frame[ref_y:ref_y+bs, ref_x:ref_x+bs]
        
        return motion_vectors, residual_frame

def visualize_results(frame1: np.ndarray, 
                    frame2: np.ndarray,
                    motion_vectors: np.ndarray,
                    residual: np.ndarray,
                    block_size: int = 16,
                    scale: float = 1.0,
                    save_path: str = "diamond_search_results.png") -> None:
    """Visualize all results together"""
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
    rows, cols = motion_vectors.shape[:2]
    y = np.arange(0, rows * block_size, block_size) + block_size//2
    x = np.arange(0, cols * block_size, block_size) + block_size//2
    X, Y = np.meshgrid(x, y)
    U = motion_vectors[:,:,0] * scale
    V = motion_vectors[:,:,1] * scale
    
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.002)
    plt.gca().invert_yaxis()
    plt.title('Motion Vectors (Diamond Search)')
    plt.grid(True, alpha=0.3)
    
    # Plot residual
    plt.subplot(2, 2, 4)
    plt.imshow(residual, cmap='gray')
    plt.title('Residual Frame')
    
    plt.tight_layout()
    
    if plt.get_backend() in ['agg', 'TkAgg']:
        plt.savefig(save_path)
        print(f"Saved results to {save_path}")
    else:
        plt.show()

def main():
    # Load video frames
    cap = cv2.VideoCapture("SampleVideo_720x480_1mb.mp4")
    ret, frame1 = cap.read()
    if not ret:
        print("Error loading video")
        return
    
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    ret, frame2 = cap.read()
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    cap.release()
    
    # Save original frames
    cv2.imwrite('frame1_ds.png', frame1)
    cv2.imwrite('frame2_ds.png', frame2)
    
    # Process frames with diamond search
    me = MotionEstimator(block_size=16, search_range=16)
    motion_vectors, residual = me.diamond_search(frame1_gray, frame2_gray)
    
    # Visualize all results together
    visualize_results(frame1_gray, frame2_gray, motion_vectors, residual,
                     save_path="diamond_search_results.png")
    
    # Save residual separately
    plt.imsave("residual_ds.png", residual, cmap='gray')
    
    print("Analysis complete. Check saved images:")
    print("- frame1_ds.png (Reference frame)")
    print("- frame2_ds.png (Current frame)")
    print("- diamond_search_results.png (All results together)")
    print("- residual_ds.png (Residual frame)")

if __name__ == "__main__":
    main()