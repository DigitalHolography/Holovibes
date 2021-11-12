/*! \file
 *
 * \brief Qt widget embeded into layout.
 */
#pragma once

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>

namespace holovibes::gui
{
class QLineEditLayout : public QHBoxLayout
{
    Q_OBJECT

  public:
    QLineEditLayout(QMainWindow* parent = nullptr, const std::string& name = "");
    ~QLineEditLayout();

    QLineEditLayout* set_name(const std::string& name);
    QLineEditLayout* set_text(const std::string& text);

    const std::string get_text();

  protected slots:
    void change_text();
  signals:
    void text_changed();

  protected:
    QLabel* label_;
    QLineEdit* line_edit_;
};
} // namespace holovibes::gui