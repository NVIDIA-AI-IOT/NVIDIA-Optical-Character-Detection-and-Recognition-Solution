from setuptools import setup, Extension
import pybind11

cpp_args = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']

sfc_module = Extension(
    'nvocdr_pybind',
    sources=['pybind.cpp'],
    include_dirs=[
        pybind11.get_include(), 
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\include',
        r'D:\03_Workspace\01_TAO\11_nvOCDR\NVIDIA-Optical-Character-Detection-and-Recognition-Solution\include',
        r'D:\03_Workspace\01_TAO\11_nvOCDR\NVIDIA-Optical-Character-Detection-and-Recognition-Solution\src',
        r'D:\03_Workspace\01_TAO\11_nvOCDR\NVIDIA-Optical-Character-Detection-and-Recognition-Solution\triton',
        r'D:\01_Software\opencv\opencv\build\include\opencv2',
        r'D:\01_Software\opencv\opencv\build\include'],
    library_dirs=[
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib\x64',
        r'D:\01_Software\opencv\opencv\build\x64\vc16\lib',
        r'D:\03_Workspace\01_TAO\11_nvOCDR\nvocdr\x64\Release'
        ],
    libraries=[
        'cudart_static',
        'nvinfer_10',
        'nvinfer_plugin_10',
        'cublas',
        'nvocdr_cpp',
        'opencv_world490'
        ],
    language='c++',
    extra_compile_args=cpp_args,
)

setup(
    name='nvocdr_pybind',
    version='1.0',
    description='Python package with nvocdr C++ extension (PyBind11)',
    ext_modules=[sfc_module],
)