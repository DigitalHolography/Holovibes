/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Contains the overloading of QGroupBox. */
#pragma once

#include <QGroupBox>
#include <QObject>

namespace holovibes
{
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
} // namespace gui
} // namespace holovibes
