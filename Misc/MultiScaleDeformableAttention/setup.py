from setuptools import setup, find_packages, Extension

setup(
    name='MultiScaleDeformableAttention',
    version='1.0',
    packages=find_packages(),
    package_data={
        'MultiScaleDeformableAttention': ['MultiScaleDeformableAttention.cpython-37m-x86_64-linux-gnu.so'],
    },
    install_requires=[
        
    ],
    python_requires='>=3.7',
)
