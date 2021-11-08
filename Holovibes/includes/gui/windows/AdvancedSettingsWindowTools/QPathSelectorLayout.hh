#pragma once

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QToolbutton>

namespace holovibes::gui
{
class QPathSelectorLayout : public QHBoxLayout
{
    Q_OBJECT
  public:
    QPathSelectorLayout(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr);
    ~QPathSelectorLayout();

    QPathSelectorLayout* setText(const std::string& text);
    QPathSelectorLayout* setName(const std::string& name);

  private slots:
    void change_folder();

  private:
    QLabel* label_;
    QLineEdit* line_edit_;
    QToolButton* browse_button_;
};
} // namespace holovibes::gui