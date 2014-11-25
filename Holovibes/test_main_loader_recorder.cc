# include "frame_file_loader.hh"
# include "compute_descriptor.hh"
# include "thread_compute.hh"
# include "recorder.hh"
# include "queue.hh"

#if 0
int main()
{
  holovibes::ComputeDescriptor comp_desc;
  comp_desc.algorithm = holovibes::ComputeDescriptor::FFT2;
  comp_desc.nsamples = 4;
  comp_desc.pindex = 1;
  comp_desc.lambda = 658e-9f;

  float z;
  int i;
  for (z = -0.095f, i = 0;
    z < -0.085f;
    z += 0.001f, ++i)
  {
    holovibes::FrameFileLoader f(
      "Mire_negative_position_160.raw",
      16,
      1024,
      1024,
      2,
      7.4f,
      camera::BIG_ENDIAN);
    comp_desc.zdistance = z;

    camera::FrameDescriptor fd = f.get_queue().get_frame_desc();
    fd.depth = 2;
    holovibes::Queue output(fd, 20);

    holovibes::ThreadCompute t(comp_desc, f.get_queue(), output);

    holovibes::Recorder r(output, "fft2_mire_" + std::to_string(i) + ".raw");
    r.record(1);
  }
  return 0;
}

#endif /* !TEST_MAIN_LOADER_RECORDER */