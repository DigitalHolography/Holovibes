#include "lightui.hh"
#include "ui_lightui.h"

namespace holovibes::gui
{
LightUI::LightUI(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::LightUI)
{
    ui->setupUi(this);
}

LightUI::~LightUI()
{
    delete ui;
}
}