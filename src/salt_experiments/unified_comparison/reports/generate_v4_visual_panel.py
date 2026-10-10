"""Render the individual reproducible V4 crops cited by ``main.tex``.

The crops deliberately reuse the documented historical Butterfly coordinates,
but obtain every displayed array from the validated V4 archive. Separate output
directories retain the two principal detector tolerances for the same image,
density, and crop. They are illustrative paired cases, not aggregate results.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[4]
CROP = (295, 265, 150, 150)  # x, y, width, height; documented legacy crop.
SCALE = 3
TOLERANCES = (0, 4)


def load(path: Path) -> Image.Image:
    if path.suffix == '.npy':
        array = np.load(path)
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert('L'))
    x, y, width, height = CROP
    crop = np.asarray(array)[y:y + height, x:x + width]
    if crop.shape != (height, width):
        raise ValueError(f'Invalid crop {CROP} for {path}: {crop.shape}')
    return Image.fromarray(crop.astype(np.uint8), mode='L').resize(
        (width * SCALE, height * SCALE), Image.Resampling.LANCZOS)


def render_noisy_with_crop_rectangle(tolerance: int) -> Path:
    """Mark the documented crop directly on the archived noisy V4 array."""
    case = ROOT / f'data/output/unifiedComparisonFinalV4/set12/tolerance_{tolerance}/extreme/05'
    output_dir = ROOT / f'figures/v4/set12_05_extreme_tau{tolerance}'
    noisy = np.load(case / 'noisy.npy')
    image = Image.fromarray(noisy.astype(np.uint8), mode='L').convert('RGB')
    x, y, width, height = CROP
    ImageDraw.Draw(image).rectangle(
        (x, y, x + width - 1, y + height - 1), outline=(0, 255, 0), width=2,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / 'noisy_with_crop_rectangle.png'
    image.save(output)
    return output


def render_tolerance(tolerance: int) -> None:
    case = ROOT / f'data/output/unifiedComparisonFinalV4/set12/tolerance_{tolerance}/extreme/05'
    output_dir = ROOT / f'figures/v4/set12_05_extreme_tau{tolerance}'
    panels = (
        ('reference.png', ROOT / 'data/input/set12/05.png'),
        ('noisy.png', case / 'noisy.npy'),
        ('nlm.png', case / 'nlm.npy'),
        ('gnlm.png', case / 'gnlm.npy'),
        ('ghnlm.png', case / 'ghnlm.npy'),
        ('ianlm.png', case / 'ianlm.npy'),
        ('median.png', case / 'median.npy'),
        ('aswmf.png', case / 'aswmf.npy'),
        ('nlmedians.png', case / 'nlmedians.npy'),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, path in panels:
        output = output_dir / filename
        load(path).save(output)
        print(output)


def main() -> None:
    for tolerance in TOLERANCES:
        render_tolerance(tolerance)
        print(render_noisy_with_crop_rectangle(tolerance))


if __name__ == '__main__':
    main()
