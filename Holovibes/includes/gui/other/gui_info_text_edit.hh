/*! \file
 *
 * \brief TODO
 */
#pragma once

#include <QObject>
#include <QString>
#include <QTimer>

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QTWidgets/QTextEdit>

namespace holovibes::gui
{
/*! \class CurvePlot
 *
 * \brief Widget wrapping for a QtChart. Used to display Chart computations.
 */
class InfoTextEdit : public QTextEdit
{
    Q_OBJECT

  public:
    InfoTextEdit(QWidget* parent = nullptr)
        : QTextEdit(parent)
    {
    }
};
} // namespace holovibes::gui