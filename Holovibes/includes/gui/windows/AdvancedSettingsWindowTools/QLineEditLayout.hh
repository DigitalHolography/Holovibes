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
/*! \class QLineEditLayout
 *
 * \brief Enhancement of input text box (line edit) system display.
 */
class QLineEditLayout : public QHBoxLayout
{
    Q_OBJECT

  public:
    /*! \brief QLineEditLayout object constructor
     *
     * \param parent the window that will embed the object
     * \param name the name to display beside the line edit
     */
    QLineEditLayout(QMainWindow* parent = nullptr, const std::string& name = "");

    /*! \brief QLineEditLayout object destructor */
    ~QLineEditLayout();

    /*! \brief Sets the name/label of the layout
     *
     * \param name the new name
     * \return QPathSelectorLayout* this, for linked initilizer purposes
     */
    QLineEditLayout* set_name(const std::string& name);

    /*! \brief Sets the selected path into the line edit box
     *
     * \param text the new text
     * \return QPathSelectorLayout* this, for linked initilizer purposes
     */
    QLineEditLayout* set_text(const std::string& text);

    /*! \brief Get the line edit's text
     *
     * \return const std::string the new text
     */
    const std::string get_text();

  signals:
    /*! \brief Calls line edit text changes*/
    void text_changed();

  protected:
    QLabel* label_;
    QLineEdit* line_edit_;
};
} // namespace holovibes::gui