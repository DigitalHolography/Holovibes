#pragma once

#include "guiafx.hh"
#include "tools.hh"

#include "camera_exception.hh"

#include "holovibes.hh"
#include "custom_exception.hh"

#include "AdvancedSettingsWindow.hh"
#include "HoloWindow.hh"
#include "SliceWindow.hh"
#include "PlotWindow.hh"
#include "MainWindow.hh"
#include "Filter2DWindow.hh"
#include "asw_mainwindow_panel.hh"

namespace holovibes
{

class UserInterface
{
  private:
    UserInterface()
    {
        std::filesystem::path holovibes_documents_path = get_user_documents_path() / "Holovibes";
        std::filesystem::create_directory(holovibes_documents_path);
        record_output_directory_ = holovibes_documents_path.string();
    }

  public:
    static inline uint window_max_size = 768;
    static inline uint auxiliary_window_max_size = 512;

  public:
    static UserInterface& instance()
    {
        static UserInterface instance{};
        return instance;
    }

    ::holovibes::gui::MainWindow* main_window = nullptr;

    std::unique_ptr<::holovibes::gui::RawWindow> xy_window = nullptr;
    std::unique_ptr<::holovibes::gui::SliceWindow> sliceXZ = nullptr;
    std::unique_ptr<::holovibes::gui::SliceWindow> sliceYZ = nullptr;
    std::unique_ptr<::holovibes::gui::RawWindow> lens_window = nullptr;
    std::unique_ptr<::holovibes::gui::RawWindow> raw_window = nullptr;
    std::unique_ptr<::holovibes::gui::Filter2DWindow> filter2d_window = nullptr;
    std::unique_ptr<::holovibes::gui::PlotWindow> plot_window_ = nullptr;
    std::unique_ptr<::holovibes::gui::AdvancedSettingsWindow> advanced_settings_window = nullptr;

    gui::ExportPanel* export_panel = nullptr;
    gui::ASWMainWindowPanel* asw_main_window_panel = nullptr;
    gui::CompositePanel* composite_panel = nullptr;
    gui::ImageRenderingPanel* image_rendering_panel = nullptr;
    gui::ImportPanel* import_panel = nullptr;
    gui::InfoPanel* info_panel = nullptr;
    gui::ViewPanel* view_panel = nullptr;

    std::string default_output_filename_{"capture"};
    std::string record_output_directory_;
    std::string file_input_directory_{"C:\\"};
    std::string batch_input_directory_{"C:\\"};

    bool is_advanced_settings_displayed = false;
    bool has_been_updated = false;

    size_t auto_scale_point_threshold_ = 100;

    // Wrapper to display correct error info
  public:
    gui::ExportPanel* const get_export_panel() const
    {
        if (export_panel != nullptr)
        {
            return export_panel;
        }
        LOG_ERROR("export_panel is nullptr");
        return nullptr;
    }
    gui::ASWMainWindowPanel* const get_asw_main_window_panel() const
    {
        if (asw_main_window_panel != nullptr)
        {
            return asw_main_window_panel;
        }
        LOG_ERROR("asw_main_window_panel is nullptr");
        return nullptr;
    }
    gui::CompositePanel* const get_composite_panel() const
    {
        if (composite_panel != nullptr)
        {
            return composite_panel;
        }
        LOG_ERROR("composite_panel is nullptr");
        return nullptr;
    }
    gui::ImageRenderingPanel* const get_image_rendering_panel() const
    {
        if (image_rendering_panel != nullptr)
        {
            return image_rendering_panel;
        }
        LOG_ERROR("image_rendering_panel is nullptr");
        return nullptr;
    }
    gui::ImportPanel* const get_import_panel() const
    {
        if (import_panel != nullptr)
        {
            return import_panel;
        }
        LOG_ERROR("import_panel is nullptr");
        return nullptr;
    }
    gui::InfoPanel* const get_info_panel() const
    {
        if (info_panel != nullptr)
        {
            return info_panel;
        }
        LOG_ERROR("info_panel is nullptr");
        return nullptr;
    }
    gui::ViewPanel* const get_view_panel() const
    {
        if (view_panel != nullptr)
        {
            return view_panel;
        }
        LOG_ERROR("view_panel is nullptr");
        return nullptr;
    }
};
} // namespace holovibes