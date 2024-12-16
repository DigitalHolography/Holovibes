#include "GUI.hh"
#include "user_interface_descriptor.hh"
#include "API.hh"

#include <regex>
#include <string>

#define UI UserInterfaceDescriptor::instance()

namespace holovibes::gui
{

void set_light_ui_mode(bool value)
{
    auto path = holovibes::settings::user_settings_filepath;
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);
    j_us["light_ui"] = value;

    std::ofstream output_file(path);
    output_file << j_us.dump(1);
}

bool is_light_ui_mode()
{
    auto path = holovibes::settings::user_settings_filepath;
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);

    return json_get_or_default(j_us, false, "light_ui");
}

QPoint getSavedHoloWindowPos()
{
    auto path = holovibes::settings::user_settings_filepath;
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);

    int x = json_get_or_default(j_us, 0, "holo window", "x");
    int y = json_get_or_default(j_us, 0, "holo window", "y");
    return QPoint(x, y);
}

QSize getSavedHoloWindowSize(ushort& width, ushort& height)
{
    auto path = holovibes::settings::user_settings_filepath;
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);

    int final_width = json_get_or_default(j_us, width, "holo window", "width");
    int final_height = json_get_or_default(j_us, height, "holo window", "height");
    return QSize(final_width, final_height);
}

void close_windows()
{
    if (UI.mainDisplay.get() != nullptr)
        UI.mainDisplay.get()->save_gui("holo window");

    UI.mainDisplay.reset(nullptr);
    UI.sliceXZ.reset(nullptr);
    UI.sliceYZ.reset(nullptr);
    UI.filter2d_window.reset(nullptr);
    UI.lens_window.reset(nullptr);
    UI.plot_window_.reset(nullptr);
    UI.raw_window.reset(nullptr);
}

void create_window(Computation window_kind, ushort window_size)
{
    auto& api = API;
    const camera::FrameDescriptor& fd = api.input.get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, window_size);

    QPoint pos = getSavedHoloWindowPos();
    QSize size = getSavedHoloWindowSize(width, height);

    if (UI.mainDisplay)
    {
        pos = UI.mainDisplay->framePosition();
        size = UI.mainDisplay->size();
        UI.mainDisplay.reset(nullptr);
    }

    if (window_kind == Computation::Raw)
    {
        UI.mainDisplay.reset(new holovibes::gui::RawWindow(pos,
                                                           size,
                                                           api.compute.get_gpu_output_queue().get(),
                                                           static_cast<float>(width) / static_cast<float>(height)));
        UI.mainDisplay->setBitshift(api.window_pp.get_raw_bitshift());
    }
    else
    {
        UI.mainDisplay.reset(new gui::HoloWindow(pos,
                                                 size,
                                                 api.compute.get_gpu_output_queue().get(),
                                                 UI.sliceXZ,
                                                 UI.sliceYZ,
                                                 static_cast<float>(width) / static_cast<float>(height)));
        UI.mainDisplay->set_is_resize(false);
        UI.mainDisplay->resetTransform();
        UI.mainDisplay->setAngle(api.window_pp.get_rotation());
        UI.mainDisplay->setFlip(api.window_pp.get_horizontal_flip());
    }

    UI.mainDisplay->setTitle(QString("XY view"));
}

void refresh_window(ushort window_size)
{
    float old_scale = 1.f;
    glm::vec2 old_translation(0.f, 0.f);
    if (UI.mainDisplay)
    {
        old_scale = UI.mainDisplay->getScale();
        old_translation = UI.mainDisplay->getTranslate();
    }

    gui::close_windows();
    gui::create_window(API.compute.get_compute_mode(), window_size);

    UI.mainDisplay->setScale(old_scale);
    UI.mainDisplay->setTranslate(old_translation[0], old_translation[1]);
}

void set_filter2d_view(bool enabled, uint auxiliary_window_max_size)
{
    if (enabled)
    {
        const camera::FrameDescriptor& fd = API.input.get_fd();
        ushort filter2d_window_width = fd.width;
        ushort filter2d_window_height = fd.height;
        get_good_size(filter2d_window_width, filter2d_window_height, auxiliary_window_max_size);

        // set positions of new windows according to the position of the
        // main GL window
        QPoint pos = UI.mainDisplay->framePosition() + QPoint(UI.mainDisplay->width() + 310, 0);
        UI.filter2d_window.reset(
            new gui::Filter2DWindow(pos,
                                    QSize(filter2d_window_width, filter2d_window_height),
                                    API.compute.get_compute_pipe()->get_filter2d_view_queue().get()));

        UI.filter2d_window->setTitle("Filter2D view");
    }
    else
        UI.filter2d_window.reset(nullptr);
}

void set_lens_view(bool enabled, uint auxiliary_window_max_size)
{
    if (enabled)
    {
        // set positions of new windows according to the position of the
        // main GL window
        QPoint pos = UI.mainDisplay->framePosition() + QPoint(UI.mainDisplay->width() + 310, 0);

        const ::camera::FrameDescriptor& fd = API.input.get_fd();
        ushort lens_window_width = fd.width;
        ushort lens_window_height = fd.height;
        get_good_size(lens_window_width, lens_window_height, auxiliary_window_max_size);

        UI.lens_window.reset(new gui::RawWindow(pos,
                                                QSize(lens_window_width, lens_window_height),
                                                API.compute.get_compute_pipe()->get_lens_queue().get(),
                                                0.f,
                                                gui::KindOfView::Lens));

        UI.lens_window->setTitle("Lens view");
    }
    else
        UI.lens_window.reset(nullptr);
}

void set_raw_view(bool enabled, uint auxiliary_window_max_size)
{
    if (enabled)
    {
        const ::camera::FrameDescriptor& fd = API.input.get_fd();
        ushort raw_window_width = fd.width;
        ushort raw_window_height = fd.height;
        get_good_size(raw_window_width, raw_window_height, auxiliary_window_max_size);

        // set positions of new windows according to the position of the main GL
        // window and Lens window
        QPoint pos = UI.mainDisplay->framePosition() + QPoint(UI.mainDisplay->width() + 310, 0);
        UI.raw_window.reset(new gui::RawWindow(pos,
                                               QSize(raw_window_width, raw_window_height),
                                               API.compute.get_compute_pipe()->get_raw_view_queue().get()));

        UI.raw_window->setTitle("Raw view");
    }
    else
        UI.raw_window.reset(nullptr);
}

void set_chart_display(bool enabled)
{
    if (enabled)
        UI.plot_window_.reset(new gui::PlotWindow(*API.compute.get_compute_pipe()->get_chart_display_queue().get(),
                                                  UI.auto_scale_point_threshold_,
                                                  "Chart"));
    else
        UI.plot_window_.reset(nullptr);
}

void set_3d_cuts_view(bool enabled, uint max_window_size)
{
    if (enabled)
    {
        auto& api = API;
        const uint window_size =
            std::max(256u, std::min(max_window_size, api.transform.get_time_transformation_size()));

        // set positions of new windows according to the position of the
        // main GL window
        QPoint xzPos = UI.mainDisplay->framePosition() + QPoint(0, UI.mainDisplay->height() + 42);
        QPoint yzPos = UI.mainDisplay->framePosition() + QPoint(UI.mainDisplay->width() + 20, 0);

        UI.sliceXZ.reset(new gui::SliceWindow(xzPos,
                                              QSize(UI.mainDisplay->width(), window_size),
                                              api.compute.get_compute_pipe()->get_stft_slice_queue(0).get(),
                                              gui::KindOfView::SliceXZ));
        UI.sliceXZ->setTitle("XZ view");
        UI.sliceXZ->setAngle(api.window_pp.get_rotation(WindowKind::XZview));
        UI.sliceXZ->setFlip(api.window_pp.get_horizontal_flip(WindowKind::XZview));

        UI.sliceYZ.reset(new gui::SliceWindow(yzPos,
                                              QSize(window_size, UI.mainDisplay->height()),
                                              api.compute.get_compute_pipe()->get_stft_slice_queue(1).get(),
                                              gui::KindOfView::SliceYZ));
        UI.sliceYZ->setTitle("YZ view");
        UI.sliceYZ->setAngle(api.window_pp.get_rotation(WindowKind::YZview));
        UI.sliceYZ->setFlip(api.window_pp.get_horizontal_flip(WindowKind::YZview));

        UI.mainDisplay->getOverlayManager().enable<gui::Cross>();

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
            UI.mainDisplay->getOverlayManager().disable(gui::SliceCross);
            UI.mainDisplay->getOverlayManager().disable(gui::Cross);
        }
    }
}

void rotate_texture()
{
    // Rotate
    double rot = API.window_pp.get_rotation();
    double new_rot = (rot == 270.f) ? 0.f : rot + 90.f;

    API.window_pp.set_rotation(new_rot);

    WindowKind window = API.view.get_current_window_type();
    if (window == WindowKind::XYview)
        UI.mainDisplay->setAngle(new_rot);
    else if (UI.sliceXZ && window == WindowKind::XZview)
        UI.sliceXZ->setAngle(new_rot);
    else if (UI.sliceYZ && window == WindowKind::YZview)
        UI.sliceYZ->setAngle(new_rot);
}

void flip_texture()
{
    bool flip = API.window_pp.get_horizontal_flip();
    API.window_pp.set_horizontal_flip(!flip);

    flip = !flip;

    WindowKind window = API.view.get_current_window_type();
    if (window == WindowKind::XYview)
        UI.mainDisplay->setFlip(flip);
    else if (UI.sliceXZ && window == WindowKind::XZview)
        UI.sliceXZ->setFlip(flip);
    else if (UI.sliceYZ && window == WindowKind::YZview)
        UI.sliceYZ->setFlip(flip);
}

void set_composite_area() { UI.mainDisplay->getOverlayManager().enable<gui::CompositeArea>(); }

void active_noise_zone() { UI.mainDisplay->getOverlayManager().enable<gui::Noise>(); }

void active_signal_zone() { UI.mainDisplay->getOverlayManager().enable<gui::Signal>(); }

void set_reticle_overlay_visible(bool value)
{
    if (value)
        UI.mainDisplay->getOverlayManager().enable<gui::Reticle>(false);
    else
        UI.mainDisplay->getOverlayManager().disable(gui::Reticle);
}

void open_advanced_settings(QMainWindow* parent, std::function<void()> callback)
{
    UI.advanced_settings_window_ = std::make_unique<gui::AdvancedSettingsWindow>(parent);
    UI.advanced_settings_window_->set_callback(callback);
}

/*!
 * \brief Extract the name from the filename
 *
 * \param[in] filePath the file name
 *
 * \return std::string the name extracted from the filename
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

    std::replace(parentPath.begin(), parentPath.end(), '/', '\\');
    UI.record_output_directory_ = std::move(parentPath);
    UI.output_filename_ = std::move(fileNameWithoutExt);
    UI.record_output_directory_ = std::move(parentPath);
    UI.output_filename_ = std::move(fileNameWithoutExt);

    return fileExt;
}

std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display() { return UI.mainDisplay; }

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_xz() { return UI.sliceXZ; }

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_yz() { return UI.sliceYZ; }

std::unique_ptr<::holovibes::gui::RawWindow>& get_lens_window() { return UI.lens_window; }

std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window() { return UI.raw_window; }

std::unique_ptr<::holovibes::gui::Filter2DWindow>& get_filter2d_window() { return UI.filter2d_window; }

} // namespace holovibes::gui