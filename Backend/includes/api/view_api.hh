/*! \file view_api.hh
 *
 * \brief Regroup all functions used to interact with the different view and view related settings.
 *
 * Views are:
 * - 3D Cuts
 * - Filter2D
 * - Chart
 * - Lens
 * - Raw
 */
#pragma once

#include "common_api.hh"
#include "enum_window_kind.hh"

namespace holovibes::api
{

/*! \class RecordApi
 *
 * \brief Regroup all functions used to interact with the different view/output (holographic, 3D cuts, filter2d, lens,
 * raw, chart) and their related settings.
 */
class ViewApi : public IApi
{

  public:
    ViewApi(const Api* api)
        : IApi(api)
    {
    }

#pragma region Focused Window

    /*! \brief Returns the type of the focused window. This setting is only useful if you use functions overload that
     * does not take a WindowKind as parameter (for contrast, log and other window specific computation).
     *
     * \return WindowKind the current window type
     */
    inline WindowKind get_current_window_type() const { return GET_SETTING(CurrentWindow); }

    /*! \brief Changes the focused window. This function is only useful if you use functions overload that does
     * not take a WindowKind as parameter (for contrast, log and other window specific computation).
     *
     * \param[in] kind the new window type
     */
    inline void change_window(const WindowKind kind) const { UPDATE_SETTING(CurrentWindow, kind); }

#pragma endregion

#pragma region 3D Cuts View

    /*! \brief Returns whether the 3D cuts view are enabled or not.
     *
     * \return bool true if enabled, false otherwise
     */
    inline bool get_cuts_view_enabled() const { return GET_SETTING(CutsViewEnabled); }

    /*! \brief Enables or Disables time transform cuts views
     *
     * \param[in] enabled true: enable, false: disable
     * \return bool true if correctly set
     */
    bool set_3d_cuts_view(bool enabled) const;

#pragma endregion

#pragma region Filter2D View

    /*! \brief Returns whether the 2D filter view is enabled or not.
     *
     * \return bool true if enabled, false otherwise
     */
    inline bool get_filter2d_view_enabled() const { return GET_SETTING(Filter2dViewEnabled); }

    /*! \brief Sets whether the 2D filter view is enabled or not.
     *
     * \param[in] value true: enable, false: disable
     */
    inline void set_filter2d_view_enabled(bool value) const { UPDATE_SETTING(Filter2dViewEnabled, value); }

    /*! \brief Adds filter2d view
     *
     * \param[in] enabled true: enable, false: disable
     */
    void set_filter2d_view(bool enabled) const;

#pragma endregion

#pragma region Chart View

    /*! \brief Returns whether the chart display is enabled or not.
     *
     * \return bool true if enabled, false otherwise
     */
    inline bool get_chart_display_enabled() const { return GET_SETTING(ChartDisplayEnabled); }

    /*! \brief Sets whether the chart display is enabled or not.
     *
     * \param[in] value true: enable, false: disable
     */
    inline void set_chart_display_enabled(bool value) const { UPDATE_SETTING(ChartDisplayEnabled, value); }

    /*! \brief Start or stop the chart display
     *
     * \param[in] enabled true: enable, false: disable
     */
    void set_chart_display(bool enabled) const;

#pragma endregion

#pragma region Lens View

    /*! \brief Returns whether the lens view is enabled or not.
     *
     * \return bool true if enabled, false otherwise
     */
    inline bool get_lens_view_enabled() const { return GET_SETTING(LensViewEnabled); }

    /*! \brief Adds or removes lens view.
     *
     * \param[in] enabled true: enable, false: disable
     */
    void set_lens_view(bool enabled) const;

#pragma endregion

#pragma region Raw View

    /*! \brief Returns whether the raw view is enabled or not.
     *
     * \return bool true if enabled, false otherwise
     */
    inline bool get_raw_view_enabled() const { return GET_SETTING(RawViewEnabled); }

    /*! \brief Adds or removes raw view

     * \param[in] enabled true: enable, false: disable
     */
    void set_raw_view(bool enabled) const;

#pragma endregion

#pragma region Last Image

    void* get_raw_last_image() const;      // get_input_queue().get()
    void* get_raw_view_last_image() const; // get_input_queue().get()
    void* get_hologram_last_image() const; // get_gpu_output_queue().get()
    void* get_lens_last_image() const;     // get_compute_pipe()->get_lens_queue().get()
    void* get_xz_last_image() const;       // get_compute_pipe()->get_stft_slice_queue(0).get()
    void* get_yz_last_image() const;       // get_compute_pipe()->get_stft_slice_queue(1).get()
    void* get_filter2d_last_image() const; // get_compute_pipe()->get_filter2d_view_queue().get()
    void* get_chart_last_image() const;    // get_compute_pipe()->get_chart_display_queue().get()

#pragma endregion

  private:
    /*! \brief Sets whether the 3D cuts view are enabled or not.
     *
     * \param[in] value true: enable, false: disable
     */
    inline void set_cuts_view_enabled(bool value) const { UPDATE_SETTING(CutsViewEnabled, value); }

    /*! \brief Sets whether the lens view is enabled or not.
     *
     * \param[in] value true: enable, false: disable
     */
    inline void set_lens_view_enabled(bool value) const { UPDATE_SETTING(LensViewEnabled, value); }

    /*! \brief Sets whether the raw view is enabled or not.
     *
     * \param[in] value true: enable, false: disable
     */
    inline void set_raw_view_enabled(bool value) const { UPDATE_SETTING(RawViewEnabled, value); }
};

} // namespace holovibes::api