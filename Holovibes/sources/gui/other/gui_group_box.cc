#include "gui_group_box.hh"

namespace gui
{
  GroupBox::GroupBox(QWidget* parent)
    : QGroupBox(parent)
  {
  }

  GroupBox::~GroupBox()
  {
  }

  void GroupBox::ShowOrHide()
  {
    if (this->isVisible())
      hide();
	else
      show();
  }
}