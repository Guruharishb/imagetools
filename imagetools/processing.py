from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np


 

def grayscale(input_path, output_path):
    img = Image.open(input_path)
    img.convert("L").save(output_path)


def resize(input_path, output_path, width, height):
    img = Image.open(input_path)
    img.resize((width, height)).save(output_path)


def rotate(input_path, output_path, angle):
    img = Image.open(input_path)
    img.rotate(angle).save(output_path)


def blur(input_path, output_path):
    img = Image.open(input_path)
    img.filter(ImageFilter.BLUR).save(output_path)


def brighten(input_path, output_path, factor):
    img = Image.open(input_path)
    enhancer = ImageEnhance.Brightness(img)
    enhancer.enhance(factor).save(output_path)


def edge_detect(input_path, output_path):
    img = Image.open(input_path)
    img.filter(ImageFilter.FIND_EDGES).save(output_path)



def inpaint(input_path, mask_path, output_path, radius=3):


    img = cv2.imread(input_path)
    mask = cv2.imread(mask_path, 0)

    result = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)

    cv2.imwrite(output_path, result)



def image_quality_score(input_path):
    img = cv2.imread(input_path, 0)

    if img is None:
        return 0

    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

    score = min(laplacian_var / 10, 100)

    return round(score, 2)


#clean pipeline

def cleanup_pipeline(input_path, output_path, resize_to=None, denoise=False):

    img = cv2.imread(input_path)

    if img is None:
        return

    if resize_to:
        img = cv2.resize(img, resize_to)

    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)

    cv2.imwrite(output_path, img)




def restore_image(input_path, output_path):

    img = cv2.imread(input_path)

    if img is None:
        return

    # denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # sharpen kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    img = cv2.filter2D(img, -1, kernel)

    # contrast
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

    cv2.imwrite(output_path, img)