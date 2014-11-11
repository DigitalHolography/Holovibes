#include "stdafx.h"

# include "frame_file_loader.hh"
# include "compute_descriptor.hh"
# include "thread_compute.hh"
# include "recorder.hh"

#if 1
int main()
{
  for (float z = -0.205f; z < -0.055f; z += 0.001f)
  {
    static int i = 0;
    holovibes::FrameFileLoader f(
      "Mire_negative_position_160.raw",
      16,
      1024,
      1024,
      2,
      5.5f,
      camera::BIG_ENDIAN);

    holovibes::ComputeDescriptor comp_desc;
    comp_desc.algorithm = holovibes::ComputeDescriptor::FFT2;
    comp_desc.nsamples = 4;
    comp_desc.pindex = 1;
    comp_desc.lambda = 686e-9f;
    comp_desc.zdistance = z;

    holovibes::ThreadCompute t(comp_desc, f.get_queue());

    holovibes::Recorder r(t.get_queue(), "fft2_mire_" + std::to_string(i) + ".raw");

    r.record(1);
    i++;
  }
  return 0;
}

#endif /* !TEST_MAIN_LOADER_RECORDER */