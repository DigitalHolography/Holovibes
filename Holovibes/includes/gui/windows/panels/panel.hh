/*! \file
 *
 * \brief Contains the overloading of QFrame.
 */
#pragma once

#include <QGroupBox>
#include <QObject>

#include <boost/property_tree/ptree.hpp>
#include <nlohmann/json.hpp>
using json = ::nlohmann::json;

namespace Ui
{
class MainWindow;
}

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

    virtual void on_notify(){};

    // #TODO Put this into constructors when .ui files exist for every panel
    virtual void init(){};

    virtual void load_gui(const json& j_us){};
    virtual void save_gui(json& j_us){};

    /*! \brief Changes Box value without triggering any signal
     *
     * \param spinBox The box to change
     * \param value The value to set
     */
    static void QSpinBoxQuietSetValue(QSpinBox* spinBox, int value);
    /*! \brief Changes Slider value without triggering any signal
     *
     * \param slider The slider to change
     * \param value The value to set
     */
    static void QSliderQuietSetValue(QSlider* slider, int value);
    /*! \brief Changes SpinBox value without triggering any signal
     *
     * \param spinBox The spinbox to change
     * \param value The value to set
     */
    static void QDoubleSpinBoxQuietSetValue(QDoubleSpinBox* spinBox, double value);

  public slots:
    /*! \brief Show or hide Panel */
    void ShowOrHide();

  protected:
    MainWindow* parent_;
    Ui::MainWindow* ui_;

  private:
    /*! \brief Recursively search the parent tree to find the MainWindow */
    MainWindow* find_main_window(QObject* widget);
};
} // namespace holovibes::gui
