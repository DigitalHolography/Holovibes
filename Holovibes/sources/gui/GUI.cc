#include "GUI.hh"
#include "user_interface_descriptor.hh"
#include "API.hh"

#include <regex>
#include <string>

#define UI UserInterfaceDescriptor::instance()

namespace holovibes::gui
{

void set_filter2d_view(bool checked, uint auxiliary_window_max_size)
{
    if (checked)
    {
        const camera::FrameDescriptor& fd = api::get_fd();
        ushort filter2d_window_width = fd.width;
        ushort filter2d_window_height = fd.height;
        get_good_size(filter2d_window_width, filter2d_window_height, auxiliary_window_max_size);

        // set positions of new windows according to the position of the
        // main GL window
        QPoint pos = UI.mainDisplay->framePosition() + QPoint(UI.mainDisplay->width() + 310, 0);
        UI.filter2d_window.reset(new gui::Filter2DWindow(pos,
                                                         QSize(filter2d_window_width, filter2d_window_height),
                                                         api::get_compute_pipe()->get_filter2d_view_queue().get()));

        UI.filter2d_window->setTitle("Filter2D view");
    }
    else
        UI.filter2d_window.reset(nullptr);
}

void set_lens_view(bool checked, uint auxiliary_window_max_size)
{
    if (checked)
    {
        // set positions of new windows according to the position of the
        // main GL window
        QPoint pos = UI.mainDisplay->framePosition() + QPoint(UI.mainDisplay->width() + 310, 0);

        const ::camera::FrameDescriptor& fd = api::get_fd();
        ushort lens_window_width = fd.width;
        ushort lens_window_height = fd.height;
        get_good_size(lens_window_width, lens_window_height, auxiliary_window_max_size);

        UI.lens_window.reset(new gui::RawWindow(pos,
                                                QSize(lens_window_width, lens_window_height),
                                                api::get_compute_pipe()->get_lens_queue().get(),
                                                0.f,
                                                gui::KindOfView::Lens));

        UI.lens_window->setTitle("Lens view");
    }
    else
        UI.lens_window.reset(nullptr);
}

void set_raw_view(bool checked, uint auxiliary_window_max_size)
{
    if (checked)
    {
        const ::camera::FrameDescriptor& fd = api::get_fd();
        ushort raw_window_width = fd.width;
        ushort raw_window_height = fd.height;
        get_good_size(raw_window_width, raw_window_height, auxiliary_window_max_size);

        // set positions of new windows according to the position of the main GL
        // window and Lens window
        QPoint pos = UI.mainDisplay->framePosition() + QPoint(UI.mainDisplay->width() + 310, 0);
        UI.raw_window.reset(new gui::RawWindow(pos,
                                               QSize(raw_window_width, raw_window_height),
                                               api::get_compute_pipe()->get_raw_view_queue().get()));

        UI.raw_window->setTitle("Raw view");
    }
    else
        UI.raw_window.reset(nullptr);
}

void set_chart_display(bool checked)
{
    if (checked)
    {
        UI.plot_window_.reset(new gui::PlotWindow(*api::get_compute_pipe()->get_chart_display_queue().get(),
                                                  UI.auto_scale_point_threshold_,
                                                  "Chart"));
    }
    else
        UI.plot_window_.reset(nullptr);
}

void set_3d_cuts_view(bool checked, uint window_size)
{
    if (checked)
    {
        window_size = std::max(256u, std::min(512u, window_size));

        // set positions of new windows according to the position of the
        // main GL window
        QPoint xzPos = UI.mainDisplay->framePosition() + QPoint(0, UI.mainDisplay->height() + 42);
        QPoint yzPos = UI.mainDisplay->framePosition() + QPoint(UI.mainDisplay->width() + 20, 0);

        UI.sliceXZ.reset(new gui::SliceWindow(xzPos,
                                              QSize(UI.mainDisplay->width(), window_size),
                                              api::get_compute_pipe()->get_stft_slice_queue(0).get(),
                                              gui::KindOfView::SliceXZ));
        UI.sliceXZ->setTitle("XZ view");
        UI.sliceXZ->setAngle(api::get_xz_rotation());
        UI.sliceXZ->setFlip(api::get_xz_horizontal_flip());

        UI.sliceYZ.reset(new gui::SliceWindow(yzPos,
                                              QSize(window_size, UI.mainDisplay->height()),
                                              api::get_compute_pipe()->get_stft_slice_queue(1).get(),
                                              gui::KindOfView::SliceYZ));
        UI.sliceYZ->setTitle("YZ view");
        UI.sliceYZ->setAngle(api::get_yz_rotation());
        UI.sliceYZ->setFlip(api::get_yz_horizontal_flip());

        UI.mainDisplay->getOverlayManager().create_overlay<gui::Cross>();

        auto holo = dynamic_cast<gui::HoloWindow*>(UI.mainDisplay.get());
        if (holo)
            holo->update_slice_transforms();
    }
    else
    {
        UI.sliceXZ.reset(nullptr);
        UI.sliceYZ.reset(nullptr);

        if (UI.mainDisplay)
        {
            UI.mainDisplay->setCursor(Qt::ArrowCursor);
            UI.mainDisplay->getOverlayManager().disable_all(gui::SliceCross);
            UI.mainDisplay->getOverlayManager().disable_all(gui::Cross);
        }
    }
}

void set_composite_area() { UI.mainDisplay->getOverlayManager().create_overlay<gui::CompositeArea>(); }

void active_noise_zone() { UI.mainDisplay->getOverlayManager().create_overlay<gui::Noise>(); }

void active_signal_zone() { UI.mainDisplay->getOverlayManager().create_overlay<gui::Signal>(); }

void open_advanced_settings(QMainWindow* parent)
{
    UI.is_advanced_settings_displayed = true;
    UI.advanced_settings_window_ = std::make_unique<::holovibes::gui::AdvancedSettingsWindow>(parent);
}

/**
 * @brief Extract the name from the filename
 *
 * @param filePath the file name
 * @return std::string the name extracted from the filename
 */
std::string getNameFromFilename(const std::string& filename)
{
    std::regex filenamePattern{R"(^\d{6}_(.*?)_?\d*$)"};
    std::smatch matches;
    if (std::regex_search(filename, matches, filenamePattern))
        return matches[1].str();
    else
        return filename; // Returning the original filename if no match is found
}

const std::string browse_record_output_file(std::string& std_filepath)
{
    // Let std::filesystem handle path normalization and system compatibility
    std::filesystem::path normalizedPath(std_filepath);

    // Using std::filesystem to derive parent path, extension, and stem directly
    std::string parentPath = normalizedPath.parent_path().string();
    std::string fileExt = normalizedPath.extension().string();
    std::string fileNameWithoutExt = getNameFromFilename(normalizedPath.stem().string());

    // Setting values in UserInterfaceDescriptor instance in a more optimized manner
    std::replace(parentPath.begin(), parentPath.end(), '/', '\\');
    UserInterfaceDescriptor::instance().record_output_directory_ = std::move(parentPath);
    UserInterfaceDescriptor::instance().output_filename_ = std::move(fileNameWithoutExt);

    return fileExt;
}

std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display() { return UI.mainDisplay; }

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_xz() { return UI.sliceXZ; }

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_yz() { return UI.sliceYZ; }

std::unique_ptr<::holovibes::gui::RawWindow>& get_lens_window() { return UI.lens_window; }

std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window() { return UI.raw_window; }

std::unique_ptr<::holovibes::gui::Filter2DWindow>& get_filter2d_window() { return UI.filter2d_window; }

} // namespace holovibes::gui