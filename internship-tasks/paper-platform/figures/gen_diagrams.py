"""Generate all paper diagrams as PNG images using Pillow"""
from PIL import Image, ImageDraw, ImageFont
import os

# === Font setup ===
FONT_CJK = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_CJK_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"

# Fallbacks
for path in [FONT_CJK, "/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(path):
        FONT_CJK = path
        break

# If TTC, need font index
def load_font(path, size, index=0):
    try:
        return ImageFont.truetype(path, size, index=index)
    except:
        try:
            return ImageFont.truetype(path, size)
        except:
            return ImageFont.load_default()

# Try multiple paths and indices for CJK
FONT_CJK_PATHS = [
    ("/System/Library/Fonts/STHeiti Light.ttc", 0),
    ("/System/Library/Fonts/PingFang.ttc", 0),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 3),
    ("/System/Library/Fonts/Supplemental/Songti.ttc", 0),
    ("/Library/Fonts/Arial Unicode.ttf", 0),
]

fnt_n = None
fnt_b = None
for path, idx in FONT_CJK_PATHS:
    if os.path.exists(path):
        try:
            fnt_n = ImageFont.truetype(path, 14, index=idx)
            fnt_b = ImageFont.truetype(path, 14, index=idx)
            print(f"Using font: {path} index={idx}")
            break
        except Exception as e:
            print(f"  Failed {path}[{idx}]: {e}")

if fnt_n is None:
    fnt_n = ImageFont.load_default()
    fnt_b = ImageFont.load_default()
    print("⚠️ Using default font")

def get_font(size, bold=False):
    for path, idx in FONT_CJK_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=idx)
            except:
                pass
    return ImageFont.load_default()

# Test
test_img = Image.new('RGB', (100, 40), 'white')
d = ImageDraw.Draw(test_img)
d.text((5,5), "测试", fill='black', font=get_font(12))
print("Font test OK")

print("✅ Font loaded, generating diagrams...")
