from setuptools import setup, find_packages

setup(
    name="imagetools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "Pillow",
        "opencv-python",
        "numpy"
    ],
    description="A simple image processing toolkit with Pillow + OpenCV features",
    author="Guruharish B",
    python_requires=">=3.7",
)