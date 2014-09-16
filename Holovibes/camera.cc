#include "camera.hh"

Camera::Camera(char* name)
{
  //FIXME: copy name into _name
}

Camera::~Camera()
{
  //Free all ressources used
  //free(_name);
}

char*
Camera::getName()
{
  return _name;
}
