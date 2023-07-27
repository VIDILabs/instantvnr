#include "tcnn_threadblock.h"
#include "tcnn_device_api.h"

/* namespace instant neural volume */
namespace vnr {
namespace tcnn_impl {

template struct DeviceNeuralNetwork<precision_t, 16>;
template struct DeviceNeuralNetwork<precision_t, 32>;
template struct DeviceNeuralNetwork<precision_t, 64>;

}
}