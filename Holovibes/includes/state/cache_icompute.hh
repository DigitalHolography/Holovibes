#pragma once

#include "parameters_handler.hh"

#include "advanced.hh"
#include "compute.hh"

namespace holovibes
{
class CacheICompute : public ParametersHandlerCache<BatchSize,
                                                    DivideConvolutionEnable,
                                                    Lambda,
                                                    DisplayRate,
                                                    InputBufferSize,
                                                    TimeStride>
{
};
} // namespace holovibes
