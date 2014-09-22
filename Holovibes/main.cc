#include <iostream>
#include <Windows.h>
#include "gl_window.hh"

int main()
{
  std::cout << "Hello World!" << std::endl;
  if (gui::GLWindow::get_instance().init("Test", 200, 200));

  return 0;
}