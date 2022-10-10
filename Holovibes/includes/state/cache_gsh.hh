#pragma once

#include "parameters_handler.hh"

#include "cache_icompute.hh"

#include "advanced.hh"
#include "compute.hh"

namespace holovibes
{
using GSHCachesToSync = CachesToSync<CacheICompute>;

class CacheGSH : public ParametersHandlerRef<CacheGSH,
                                             GSHCachesToSync,

                                             BatchSize,
                                             DivideConvolutionEnable,
                                             Lambda,
                                             DisplayRate,
                                             InputBufferSize,
                                             TimeStride>
{
};
} // namespace holovibes
