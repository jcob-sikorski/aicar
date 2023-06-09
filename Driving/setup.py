from setuptools import setup

setup(
    name="driving",
    version='0.0.1',
    install_requires=[
        'gym',
        'pybullet',
        'numpy',
        'matplotlib',
        'opencv-python',  # for cv2
        'sb3_contrib'  # for RecurrentPPO
    ]
)
