/*! \file
 *
 * \brief Contains the overloading of QFrame.
 */
#pragma once

#include <QGroupBox>
#include <QObject>

namespace holovibes::gui
{
class MainWindow;

/*! \class Panel
 *
 * \brief QGroupBox overload, used to hide and show parts of the GUI.
 */
class Panel : public QGroupBox
{
    Q_OBJECT

  public:
    /*! \brief Panel constructor
     * \param parent Qt parent
     */
    Panel(QWidget* parent = nullptr);
    /*! \brief Panel destructor */
    ~Panel();

  public slots:
    /*! \brief Show or hide Panel */
    void ShowOrHide();

  protected:
    /*! \brief Recursively search the parent tree to find the MainWindow */
    MainWindow* find_main_window(QObject* widget);
};
} // namespace holovibes::gui
