#include "base.h"
#include "nvocdr.h"
namespace nvocdr
{
//  BaseProcessor
template<typename Param>
bool BaseProcessor<Param>::infer(bool sync_input, const cudaStream_t& stream) {
    for(auto & [_, engine]: mEngines) {
        if(sync_input) {
            engine->syncMemory(true, true, stream);
        }
        engine->infer(stream);
    }
    return true;
}
template class BaseProcessor<nvOCRParam>;
template class BaseProcessor<nvOCDParam>;
} // namespace nvocdr
