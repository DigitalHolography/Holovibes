#pragma once

#include "custom_parameter.hh"
#include "micro_cache_tmp.hh"

NEW_INITIALIZED_MICRO_CACHE(CompositeCache,
                            (CompositeKind, composite_kind, CompositeKind::RGB),
                            (bool, composite_auto_weights, false),
                            (Composite_RGB, rgb, Composite_RGB{}),
                            (Composite_HSV, hsv, Composite_HSV{}));

namespace holovibes
{

using DisplayRate = FloatParameter<30, "display_rate">;
using InputBufferSize = UIntParameter<512, "input_buffer_size">;
using OutputBufferSize = UIntParameter<256, "output_buffer_size">;
using RecordBufferSize = UIntParameter<1024, "record_buffer_size">;

using CompositeCache = MicroCacheTmp<DisplayRate,
                                     InputBufferSize,
                                     OutputBufferSize,
                                     RecordBufferSize,
                                     ContrastLowerThreshold,
                                     ContrastUpperThreshold,
                                     RawBitshift,
                                     RenormConstant,
                                     CutsContrastPOffset>;

} // namespace holovibes
