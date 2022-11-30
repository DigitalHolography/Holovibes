/*! \file
 *
 * \brief Contains the overloading of QFrame.
 */
#pragma once

#include <QFrame>
#include <QObject>

namespace holovibes::gui
{
/*! \class Frame
 *
 * \brief QFrame overload, used to hide and show parts of the GUI.
 */
class Frame : public QFrame
{
    Q_OBJECT

  public:
    /*! \brief Frame constructor
     * \param parent Qt parent
     */
    Frame(QWidget* parent = nullptr);
    /*! \brief Frame destructor */
    ~Frame();

  public slots:
    /*! \brief Show or hide Frame */
    void ShowOrHide();
};
} // namespace holovibes::gui
