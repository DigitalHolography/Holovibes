#ifndef GUI_GROUP_BOX_HH_
# define GUI_GROUP_BOX_HH_

# include <QGroupBox>

namespace gui
{
  /*! \class GroupBox
  **
  ** QGroupBox overload, used to hide and show parts of the GUI.
  */
  class GroupBox : public QGroupBox
  {
    Q_OBJECT

  public:
    /*! \brief GroupBox constructor
    ** \param parent Qt parent 
    */
    GroupBox(QWidget* parent = 0);
    /*! \brief GroupBox destructor */
    ~GroupBox();

  public slots:
    /*! \brief Show or hide GroupBox */
    void ShowOrHide();
  };
}

#endif /* !GUI_GROUP_BOX_HH_ */