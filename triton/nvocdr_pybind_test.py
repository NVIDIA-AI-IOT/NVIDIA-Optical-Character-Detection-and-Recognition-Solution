import os



os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib\x64')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin')
os.add_dll_directory(r'D:\01_Software\opencv\opencv\build\x64\vc16\lib')
os.add_dll_directory(r'D:\01_Software\opencv\opencv\build\x64\vc16\bin')


import nvocdr_pybind
print('load nvocdr successfully')