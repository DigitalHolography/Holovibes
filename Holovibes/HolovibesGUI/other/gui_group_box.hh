/*! \file
 *
 * \brief Contains the overloading of QGroupBox.
 */
#pragma once

#include <QGroupBox>
#include <QObject>

namespace holovibes::gui
{
/*! \class GroupBox
 *
 * \brief QGroupBox overload, used to hide and show parts of the GUI.
 */
class GroupBox : public QGroupBox
{
    Q_OBJECT

  public:
    /*! \brief GroupBox constructor
     * \param parent Qt parent
     */
    GroupBox(QWidget* parent = nullptr);
    /*! \brief GroupBox destructor */
    ~GroupBox();

  public slots:
    /*! \brief Show or hide GroupBox */
    void ShowOrHide();
};
} // namespace holovibes::gui
