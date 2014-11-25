#ifndef GUI_GROUP_BOX_HH_
# define GUI_GROUP_BOX_HH_

# include <QGroupBox>

namespace gui
{
  class GroupBox : public QGroupBox
  {
    Q_OBJECT

  public:
    GroupBox(QWidget* parent = 0);
    ~GroupBox();

  public slots:
    void ShowOrHide();
  };
}

#endif /* !GUI_GROUP_BOX_HH_ */