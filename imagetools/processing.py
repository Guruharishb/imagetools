from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


PathLike = Union[str, Path]


def _validate_input(path: PathLike) -> Path:
    """Validate that an input file exists."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    return path


def _validate_output(path: PathLike) -> Path:
    """Return an output path and create its parent directory if necessary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def grayscale(input_path: PathLike, output_path: PathLike) -> None:
    """Convert an image to grayscale and save the result."""
    input_path = _validate_input(input_path)
    output_path = _validate_output(output_path)

    with Image.open(input_path) as img:
        img.convert("L").save(output_path)


def resize(
    input_path: PathLike,
    output_path: PathLike,
    width: int,
    height: int,
) -> None:
    """Resize an image to the requested dimensions."""
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")

    input_path = _validate_input(input_path)
    output_path = _validate_output(output_path)

    with Image.open(input_path) as img:
        resized = img.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )
        resized.save(output_path)


def rotate(
    input_path: PathLike,
    output_path: PathLike,
    angle: float,
) -> None:
    """Rotate an image by the given angle in degrees."""
    input_path = _validate_input(input_path)
    output_path = _validate_output(output_path)

    with Image.open(input_path) as img:
        rotated = img.rotate(angle, expand=True)
        rotated.save(output_path)


def blur(input_path: PathLike, output_path: PathLike) -> None:
    """Apply a blur filter to an image."""
    input_path = _validate_input(input_path)
    output_path = _validate_output(output_path)

    with Image.open(input_path) as img:
        img.filter(ImageFilter.BLUR).save(output_path)


def brighten(
    input_path: PathLike,
    output_path: PathLike,
    factor: float,
) -> None:
    """Adjust image brightness using the supplied factor."""
    if factor < 0:
        raise ValueError("Brightness factor must be non-negative.")

    input_path = _validate_input(input_path)
    output_path = _validate_output(output_path)

    with Image.open(input_path) as img:
        enhancer = ImageEnhance.Brightness(img)
        enhancer.enhance(factor).save(output_path)


def edge_detect(input_path: PathLike, output_path: PathLike) -> None:
    """Detect edges in an image using Pillow's FIND_EDGES filter."""
    input_path = _validate_input(input_path)
    output_path = _validate_output(output_path)

    with Image.open(input_path) as img:
        img.filter(ImageFilter.FIND_EDGES).save(output_path)


def inpaint(
    input_path: PathLike,
    mask_path: PathLike,
    output_path: PathLike,
    radius: float = 3,
) -> None:
    """Remove masked regions using OpenCV Telea inpainting."""
    if radius <= 0:
        raise ValueError("Inpainting radius must be positive.")

    input_path = _validate_input(input_path)
    mask_path = _validate_input(mask_path)
    output_path = _validate_output(output_path)

    image = cv2.imread(str(input_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to read image: {input_path}")

    if mask is None:
        raise ValueError(f"Unable to read mask: {mask_path}")

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask must have the same dimensions.")

    result = cv2.inpaint(
        image,
        mask,
        radius,
        cv2.INPAINT_TELEA,
    )

    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"Failed to save output image: {output_path}")


def image_quality_score(input_path: PathLike) -> float:
    """Calculate a simple sharpness-based image quality score."""
    input_path = _validate_input(input_path)

    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(f"Unable to read image: {input_path}")

    laplacian_variance = cv2.Laplacian(
        image,
        cv2.CV_64F,
    ).var()

    score = min(laplacian_variance / 10, 100)

    return round(float(score), 2)


def cleanup_pipeline(
    input_path: PathLike,
    output_path: PathLike,
    resize_to: Optional[Tuple[int, int]] = None,
    denoise: bool = False,
) -> None:
    """Run resizing, optional denoising, and contrast adjustment."""
    input_path = _validate_input(input_path)
    output_path = _validate_output(output_path)

    if resize_to is not None:
        width, height = resize_to

        if width <= 0 or height <= 0:
            raise ValueError(
                "Resize dimensions must be positive integers."
            )

    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError(f"Unable to read image: {input_path}")

    if resize_to is not None:
        image = cv2.resize(image, resize_to)

    if denoise:
        image = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            10,
            10,
            7,
            21,
        )

    image = cv2.convertScaleAbs(
        image,
        alpha=1.1,
        beta=10,
    )

    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to save output image: {output_path}")


def restore_image(
    input_path: PathLike,
    output_path: PathLike,
) -> None:
    """Restore an image using denoising, sharpening, and contrast."""
    input_path = _validate_input(input_path)
    output_path = _validate_output(output_path)

    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError(f"Unable to read image: {input_path}")

    # Denoise
    image = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        10,
        10,
        7,
        21,
    )

    # Sharpen
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    image = cv2.filter2D(
        image,
        -1,
        kernel,
    )

    # Improve contrast
    image = cv2.convertScaleAbs(
        image,
        alpha=1.2,
        beta=10,
    )

    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to save output image: {output_path}")
