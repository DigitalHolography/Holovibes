#include "stdafx.h"
#include "main_window.hh"

Window::Window(QWidget *parent)
  : QMainWindow(parent)
{
  ui.setupUi(this);
}

Window::~Window()
{

}