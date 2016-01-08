#include "compute_descriptor.hh"

namespace holovibes
{
  ComputeDescriptor& ComputeDescriptor::operator=(const ComputeDescriptor& cd)
  {
    algorithm = cd.algorithm.load();
    nsamples = cd.nsamples.load();
    pindex = cd.pindex.load();
    lambda = cd.lambda.load();
    zdistance = cd.zdistance.load();
    view_mode = cd.view_mode.load();
    log_scale_enabled = cd.log_scale_enabled.load();
    shift_corners_enabled = cd.shift_corners_enabled.load();
    contrast_enabled = cd.contrast_enabled.load();
    vibrometry_enabled = cd.vibrometry_enabled.load();
    contrast_min = cd.contrast_min.load();
    contrast_max = cd.contrast_max.load();
    vibrometry_q = cd.vibrometry_q.load();
    autofocus_size = cd.autofocus_size.load();

    return *this;
  }
}