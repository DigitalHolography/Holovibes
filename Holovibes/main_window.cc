#include "stdafx.h"
#include "main_window.hh"

namespace gui
{
  MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
  {
    ui.setupUi(this);
  }

  MainWindow::~MainWindow()
  {
  }
}