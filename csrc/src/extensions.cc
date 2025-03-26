#include "gpu_ops.cuh"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>

namespace ffi = xla::ffi;
namespace nb = nanobind;

ffi::Error AddElementImpl(cudaStream_t stream, float scaler, ffi::Buffer<ffi::DataType::F32> x,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> y) {
    add_element(scaler, x.typed_data(), y->typed_data(), x.element_count(), stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(AddElement, AddElementImpl,
                              ffi::Ffi::Bind()
                                      .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                      .Attr<float>("scaler")  // scaler
                                      .Arg<ffi::Buffer<ffi::DataType::F32>>()
                                      .Ret<ffi::Buffer<ffi::DataType::F32>>()  // y
);

template <typename T>
nb::capsule EncapsulateFfiCall(T *fn) {
    // This check is optional, but it can be helpful for avoiding invalid
    // handlers.
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

nb::dict Registrations() {
    nb::dict d;
    d["add_element"] = EncapsulateFfiCall(AddElement);
    return d;
}

NB_MODULE(gpu_ops, m) { m.def("registrations", &Registrations); }
