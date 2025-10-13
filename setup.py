from setuptools import setup, find_packages

setup(
    name="easytune",
    version="0.1.0",
    description="Lightweight fine-tuning for image/text embedding models",
    author="EasyTune Contributors",
    packages=find_packages(exclude=("tests", "examples")),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "datasets>=2.14",
        "tqdm>=4.65",
        "pillow>=9.0",
        "numpy>=1.23",
        "pandas>=1.5",
        "torchvision>=0.15",
    ],
    python_requires=">=3.8",
)
