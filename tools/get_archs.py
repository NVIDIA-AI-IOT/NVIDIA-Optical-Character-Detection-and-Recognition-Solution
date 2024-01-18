import ctypes

driver = ctypes.cdll.LoadLibrary("libcuda.so")
flags = ctypes.c_int(0)

driver.cuInit(flags)
count = ctypes.c_int(0)
driver.cuDeviceGetCount(ctypes.pointer(count))

flag = f""
device_arch_list = set()
for device in range(count.value):
    major = ctypes.c_int(0)
    minor = ctypes.c_int(0)
    driver.cuDeviceComputeCapability(
            ctypes.pointer(major),
            ctypes.pointer(minor),
            device)
    device_arch_list.add(f"{major.value}{minor.value}")

device_arch_list = list(device_arch_list)
output=f"{device_arch_list[0]}"
for arch in device_arch_list[1:]:
    output += f";{arch}"
print(output, end=None)
