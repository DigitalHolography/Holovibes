#include "stdafx.h"

#include "options_parser.hh"
#include "holovibes.hh"

#include "camera.hh"
#include "camera_pixelfly.hh"
#include "queue.hh"
#include "test.cuh"

/*int main(int argc, const char* argv[])
{
  holovibes::OptionsDescriptor opts;
  holovibes::OptionsParser opts_parser(opts);
  opts_parser.parse(argc, argv);

  holovibes::Holovibes h(opts.camera);

  h.init_capture(opts.queue_size);
  if (opts.is_gl_window_enabled)
    h.init_display(opts.gl_window_width, opts.gl_window_height);
  if (opts.is_recorder_enabled)
h.init_recorder(opts.recorder_filepath, opts.recorder_set_size, opts.recorder_n_img);


  h.init_compute();



  h.dispose_compute();
  h.dispose_display();
  h.dispose_recorder();
  h.dispose_capture();
//getchar();
  return 0;
}*/



void load_images()
{}
int main(int argc, const char* argv[])
{
  //camera::Camera *cam = new camera::CameraPixelfly();
  //cam->init_camera();
  //cam->start_acquisition();

  camera::FrameDescriptor desc;
  desc.depth = 1;
  desc.height = 2048;
  desc.width = 2048;
  desc.pixel_size = 1;
  FILE *fd;

  int images_2_read = 16;
  fopen_s(&fd, "2phase_rawfrog.raw", "r+b");
  unsigned char *img =(unsigned char*) malloc(16 * 2048 * 2048);
  fread((void*)img, 2048 * 2048, 16, fd);



  //img2disk("testload.raw", img, 2048 * 2048 * 16);



  holovibes::Queue *q = new holovibes::Queue(desc, 50);
  for (int i = 0; i < images_2_read; i++)
  {
    q->enqueue((void*) &img[i * desc.width * desc.height * desc.depth]);
  }

  test_fft(2, q);
  getchar();
  //test_16(10,q);
  //getchar();
  /*int nbimages = 10;
  for (int i = 0; i < nbimages; i++)
  {
    q->enqueue(cam->get_frame());
  }
  test_16(3, q);
  test_fft(4, q);
  getchar();*/


}
