import cv2
import pytesseract
from src.utils.ocr_text import TesseractEngine

engine = TesseractEngine()

test_cases = [
    ('test_normal.png', 'Normal (black on white)'),
    ('test_inverted.png', 'Inverted (white on black)'),
    ('test_noisy.png', 'Noisy'),
    ('test_lowcontrast.png', 'Low Contrast'),
]

print('='*70)
print('CURRENT IMPLEMENTATION TEST')
print('='*70)

for filename, desc in test_cases:
    img = cv2.imread(f'examples/sample_pages/{filename}')
    if img is None:
        print(f'{desc}: FAILED TO LOAD')
        continue
    
    result = engine.recognize(img)
    text = result.text.strip().replace('\n', ' ')
    
    # Check quality
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    print(f'\n{desc}:')
    print(f'  Laplacian: {laplacian:.1f}, Confidence: {result.confidence:.1%}')
    print(f'  Result: "{text[:50]}"')
    
    # Expected text check
    if 'Normal' in filename and 'Normal' in text:
        print('  Status: ✓ PASS')
    elif 'Inverted' in filename and 'Inverted' in text:
        print('  Status: ✓ PASS')
    elif 'noisy' in filename and 'Normal' in text:
        print('  Status: ✓ PASS')
    elif 'lowcontrast' in filename and ('Low' in text or 'Contrast' in text):
        print('  Status: ✓ PASS')
    else:
        print('  Status: ✗ FAIL')
