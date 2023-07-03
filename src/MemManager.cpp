#include "MemManager.h"

using namespace nvocdr;


BufferManager::BufferManager()
{
    mDeviceBuffer.clear();
    mHostBuffer.clear();
}


int
BufferManager::initDeviceBuffer(const size_t data_size, const size_t item_size)
{
    int index = mDeviceBuffer.size();
    mDeviceBuffer.emplace_back(DeviceBuffer(data_size, item_size));
    return index;
}

int
BufferManager::initHostBuffer(const size_t data_size, const size_t item_size)
{
    int index = mHostBuffer.size();
    mHostBuffer.emplace_back(HostBuffer(data_size, item_size));
    return index;
}