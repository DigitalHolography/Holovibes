#include "lightui.hh"
#include "MainWindow.hh"
#pragma warning(push, 0)
#include "ui_lightui.h"
#pragma warning(pop)

namespace holovibes::gui
{
LightUI::LightUI(QWidget *parent, MainWindow* main_window)
    : QDialog(parent)
    , ui_(new Ui::LightUI), main_window_(main_window)
{
    ui_->setupUi(this);
}

LightUI::~LightUI()
{
    delete ui_;
}
}