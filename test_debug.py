import cv2
import numpy as np
from src.utils.ocr_text import TesseractEngine

h, w = 200, 500

# Heavy S&P noise
normal = np.ones((h, w), dtype=np.uint8) * 255
cv2.putText(normal, 'Heavy Noise Test', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
heavy_noise = normal.copy()
num_salt = int(0.1 * h * w)
coords = [np.random.randint(0, i-1, num_salt) for i in (h, w)]
heavy_noise[coords[0], coords[1]] = 0
coords = [np.random.randint(0, i-1, num_salt) for i in (h, w)]
heavy_noise[coords[0], coords[1]] = 255

# Uneven lighting
gradient = np.zeros((h, w), dtype=np.uint8)
for i in range(w):
    gradient[:, i] = int(50 + (200 * i / w))
cv2.putText(gradient, 'Uneven Light', (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)

# Check Laplacian for both
gray1 = heavy_noise
gray2 = gradient

lap1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
lap2 = cv2.Laplacian(gray2, cv2.CV_64F).var()

print(f'Heavy S&P Noise - Laplacian: {lap1:.1f} ({"CLEAN" if lap1 > 500 else "DEGRADED"})')
print(f'Uneven Lighting - Laplacian: {lap2:.1f} ({"CLEAN" if lap2 > 500 else "DEGRADED"})')

# The noise creates HIGH laplacian (lots of edges from noise pixels!)
# We need additional checks

# Check noise level using local variance
def estimate_noise(img):
    """Estimate noise using median absolute deviation."""
    H, W = img.shape
    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]
    sigma = np.sum(np.sum(np.abs(cv2.filter2D(img.astype(np.float64), -1, np.array(M)))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))
    return sigma

noise1 = estimate_noise(gray1)
noise2 = estimate_noise(gray2)

print(f'Heavy S&P Noise - Noise estimate: {noise1:.1f}')
print(f'Uneven Lighting - Noise estimate: {noise2:.1f}')

# Check contrast uniformity (std of local means)
def check_uneven_lighting(img, block_size=32):
    """Check if lighting is uneven by comparing local means."""
    h, w = img.shape
    means = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = img[y:y+block_size, x:x+block_size]
            means.append(block.mean())
    return np.std(means)

uneven1 = check_uneven_lighting(gray1)
uneven2 = check_uneven_lighting(gray2)

print(f'Heavy S&P Noise - Lighting variance: {uneven1:.1f}')
print(f'Uneven Lighting - Lighting variance: {uneven2:.1f}')
