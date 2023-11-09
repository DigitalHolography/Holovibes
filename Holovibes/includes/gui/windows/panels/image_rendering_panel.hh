/*! \file
 *
 * \brief File containing methods, attributes and slots related to the Image Rendering panel
 */
#pragma once

#include "panel.hh"
#include "Filter2DWindow.hh"

namespace holovibes::gui
{
class MainWindow;

/*! \class ImageRenderingPanel
 *
 * \brief Class representing the Image Rendering Panel in the GUI
 */
class ImageRenderingPanel : public Panel
{
    Q_OBJECT

  public:
    ImageRenderingPanel(QWidget* parent = nullptr);
    ~ImageRenderingPanel();

    void init() override;
    void on_notify() override;

    void load_gui(const json& j_us) override;
    void save_gui(json& j_us) override;

    std::unique_ptr<Filter2DWindow> filter2d_window = nullptr;

  public slots:
    /*! \brief Set image mode either to raw or hologram mode
     *
     * Check if Camera has been enabled, then create a new GuiGLWindow keeping
     * its old position and size if it was previously opened, set visibility
     * and call notify().
     *
     * \param value true for raw mode, false for hologram mode.
     */
    void set_image_mode(int mode);

    /*! \brief Modifies batch size from ui value */
    void update_batch_size();
    /*! \brief Modifies time transformation stride size from ui value */
    void update_time_stride();

    /*! \brief Applies or removes 2d filter on output display
     *
     * \param checked true: enable, false: disable
     */
    void set_filter2d(bool checked);
    /*! \brief adds or removes filter 2d view
     *
     * \param checked true: enable, false: disable
     */
    void update_filter2d_view(bool checked);
    /*! \brief Modifies filter2d n1 (first value)
     *
     * \param n The new filter2d n1 value
     */
    void set_filter2d_n1(int n);
    /*! \brief Modifies filter2d n2 (second value)
     *
     * \param n The new filter2d n2 value
     */
    void set_filter2d_n2(int n);
    
    /*! \brief Modifies input filter
     *
     * \param value The new filter to apply
     */
    void update_input_filter(const QString& value);

    /*! \brief Refreshed the input filter iff one was passed before
     */
    void refresh_input_filter()

    /*! \brief Modifies space transform calculation
     *
     * \param value The new space transform to apply
     */
    void set_space_transformation(const QString& value);
    /*! \brief Modifies time transform calculation
     *
     * \param value The new time transform to apply
     */
    void set_time_transformation(const QString& value);

    /*! \brief Changes the time transformation size from ui value */
    void set_time_transformation_size();
    /*! \brief Modifies wave length (lambda)
     *
     * \param value The new value of lambda
     */
    void set_wavelength(double value);
    /*! \brief Modifies z from ui value
     *
     * \param value The new value of z
     */
    void set_z(double value);
    /*! \brief Increment z by 1 on key shortcut */
    void increment_z();
    /*! \brief Decrement z by 1 on key shortcut */
    void decrement_z();

    /*! \brief Enable the convolution mode
     *
     * \param enable true: enable, false: disable
     */
    void set_convolution_mode(const bool enable);
    /*! \brief Modifies convolution kernel
     *
     * \param value The new kernel to apply
     */
    void update_convo_kernel(const QString& value);
    /*! \brief Enables the divide convolution mode
     *
     * \param value true: enable, false: disable
     */
    void set_divide_convolution(const bool value);

    /*!
     * \brief Sets the z step
     *
     * \param value the new value
     */
    void set_z_step(double value);

    /*!
     * \brief Gets the z step
     *
     * \return double the current z step
     */
    double get_z_step();

  private:
    QShortcut* z_up_shortcut_;
    QShortcut* z_down_shortcut_;

  public:
    double z_step_ = 0.005f;
};
} // namespace holovibes::gui
