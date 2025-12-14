import cv2
import numpy as np
from src.utils.ocr_text import TesseractEngine

engine = TesseractEngine()

h, w = 200, 500

print('EXTREME SCENARIO TESTS:')
print('='*70)

# 1. Heavy salt-and-pepper noise
normal = np.ones((h, w), dtype=np.uint8) * 255
cv2.putText(normal, 'Heavy Noise Test', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
heavy_noise = normal.copy()
num_salt = int(0.1 * h * w)
coords = [np.random.randint(0, i-1, num_salt) for i in (h, w)]
heavy_noise[coords[0], coords[1]] = 0
coords = [np.random.randint(0, i-1, num_salt) for i in (h, w)]
heavy_noise[coords[0], coords[1]] = 255

result = engine.recognize(cv2.cvtColor(heavy_noise, cv2.COLOR_GRAY2BGR))
status = "PASS" if "noise" in result.text.lower() else "FAIL"
print(f'Heavy S&P Noise:   {status} | {result.text.strip()[:40]}')

# 2. Very low contrast (5% difference)
vlow = np.ones((h, w), dtype=np.uint8) * 130
cv2.putText(vlow, 'Very Low Contrast', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 120, 3)
result = engine.recognize(cv2.cvtColor(vlow, cv2.COLOR_GRAY2BGR))
status = "PASS" if "contrast" in result.text.lower() else "FAIL"
print(f'Very Low Contrast: {status} | {result.text.strip()[:40]}')

# 3. Uneven lighting (gradient background)
gradient = np.zeros((h, w), dtype=np.uint8)
for i in range(w):
    gradient[:, i] = int(50 + (200 * i / w))
cv2.putText(gradient, 'Uneven Light', (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
result = engine.recognize(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR))
status = "PASS" if "light" in result.text.lower() or "uneven" in result.text.lower() else "FAIL"
print(f'Uneven Lighting:   {status} | {result.text.strip()[:40]}')

# 4. Gaussian blur (simulating out of focus)
blurred = cv2.GaussianBlur(normal, (15, 15), 0)
result = engine.recognize(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR))
status = "PASS" if "noise" in result.text.lower() else "FAIL"
print(f'Gaussian Blur:     {status} | {result.text.strip()[:40]}')

# 5. Mixed: inverted + noise
inverted_noisy = np.zeros((h, w), dtype=np.uint8)
cv2.putText(inverted_noisy, 'Inverted Noisy', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
noise = np.random.randint(0, 30, (h, w), dtype=np.uint8)
inverted_noisy = cv2.add(inverted_noisy, noise)
result = engine.recognize(cv2.cvtColor(inverted_noisy, cv2.COLOR_GRAY2BGR))
status = "PASS" if "inverted" in result.text.lower() or "noisy" in result.text.lower() else "FAIL"
print(f'Inverted + Noise:  {status} | {result.text.strip()[:40]}')

print('='*70)
