/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include <QDragEnterEvent>
#include <QLineEdit>
#include <QObject>
#include <QWidget>

namespace holovibes::gui
{
class Drag_drop_lineedit : public QLineEdit
{
    Q_OBJECT
  public:
    Drag_drop_lineedit(QWidget* parent = nullptr);

  public slots:
    void dropEvent(QDropEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* e) override
    {
        e->acceptProposedAction();
    }
};
} // namespace holovibes::gui
