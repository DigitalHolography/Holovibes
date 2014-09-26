#include <iostream>
#include <fstream>
#include <cstdio>
#include "pike_camera.hh"

int main()
{
  camera::PikeCamera cam("");
  bool is_init = cam.init_camera();

  if (is_init)
  {
    std::cout << cam.get_name() << std::endl;
    cam.start_acquisition();

    FGFRAME* frame = (FGFRAME*) cam.get_frame();

    // Writting to file
    if (frame != nullptr)
    {
      std::ofstream stream;
      stream.open("test.raw");
      if (frame->pData)
        stream.write((char*)frame->pData, frame->Length);
      else
        std::cout << "frame pdata null" << std::endl;
      stream.close();
    }

    cam.stop_acquisition();
    cam.shutdown_camera();
  }
  else
    std::cout << "Camera not initialized" << std::endl;

  getchar();
  return 0;
}