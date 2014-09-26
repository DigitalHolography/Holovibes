#include <iostream>
#include <fstream>
#include <cstdio>
#include "pike_camera.hh"
#include "queue.hh"
#include "recorder.hh"

int main()
{
  camera::PikeCamera cam("");
  bool is_init = cam.init_camera();

  if (is_init)
  {
    std::cout << cam.get_name() << std::endl;
    cam.start_acquisition();

    unsigned int imgs = 200;
    queue::Queue queue(1600 * 1200, 100);
    recorder::Recorder recorder(&queue, "test.raw", 10);

    FGFRAME* frame;
    for (int i = 0; i < imgs; ++i)
    {
      std::cout << "Img " << i << std::endl;
      frame = (FGFRAME*)cam.get_frame();
      queue.enqueue(frame->pData);
      recorder.record();
    }

    cam.stop_acquisition();
    cam.shutdown_camera();
  }
  else
    std::cout << "Camera not initialized" << std::endl;

  getchar();
  return 0;
}