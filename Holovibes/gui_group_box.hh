#pragma once

# include <QGroupBox>

namespace gui
{
  /*! \brief QGroupBox overload, used to hide and show parts of the GUI. */
  class GroupBox : public QGroupBox
  {
    Q_OBJECT

  public:
    /*! \brief GroupBox constructor
    ** \param parent Qt parent
    */
    GroupBox(QWidget* parent = nullptr);
    /*! \brief GroupBox destructor */
    ~GroupBox();

    public slots:
    /*! \brief Show or hide GroupBox */
    void ShowOrHide();
  };
}