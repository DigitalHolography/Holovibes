#pragma once

// without namespace
#include "tools.hh"

#include "enum_record_mode.hh"

// namespace camera
#include "camera_exception.hh"

// namespace holovibes
#include "holovibes.hh"
#include "custom_exception.hh"

// namespace gui
#include "AdvancedSettingsWindow.hh"
#include "HoloWindow.hh"
#include "SliceWindow.hh"
#include "PlotWindow.hh"
#include "Filter2DWindow.hh"
#include "ui_mainwindow.h"
#include "logger.h"

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
    static constexpr uint window_max_size = 768;
    static constexpr uint auxiliary_window_max_size = 512;

  public:
    static UserInterface& instance()
    {
        static UserInterface instance{};
        return instance;
    }

    std::unique_ptr<::holovibes::gui::RawWindow> main_display = nullptr;
    std::unique_ptr<::holovibes::gui::SliceWindow> sliceXZ = nullptr;
    std::unique_ptr<::holovibes::gui::SliceWindow> sliceYZ = nullptr;
    std::unique_ptr<::holovibes::gui::RawWindow> lens_window = nullptr;
    std::unique_ptr<::holovibes::gui::RawWindow> raw_window = nullptr;
    std::unique_ptr<::holovibes::gui::Filter2DWindow> filter2d_window = nullptr;
    std::unique_ptr<::holovibes::gui::PlotWindow> plot_window_ = nullptr;
    std::unique_ptr<::holovibes::gui::AdvancedSettingsWindow> advanced_settings_window = nullptr;

    ExportPanel* export_panel = nullptr;
    ASWMainWindowPanel* asw_main_window_panel = nullptr;
    CompositePanel* composite_panel = nullptr;
    ImageRenderingPnale* image_rendering_panel = nullptr;
    ImportPanel* import_panel = nullptr;
    InfoPanel* info_panel = nullptr;
    ViewPanel* view_panel = nullptr;

    std::string default_output_filename_{"capture"};
    std::string record_output_directory_;
    std::string file_input_directory_{"C:\\"};
    std::string batch_input_directory_{"C:\\"};

    bool is_advanced_settings_displayed = false;
    bool has_been_updated = false;

    size_t auto_scale_point_threshold_ = 100;

    // Wrapper to display correct error info
  public:
    auto* const get_export_panel() const
    {
        if (export_panel != nullptr)
        {
            return export_panel;
        }
        LOG_ERROR(main, "export_panel is nullptr");
        return nullptr;
    }
    auto* const get_asw_main_window_panel() const
    {
        if (asw_main_window_panel != nullptr)
        {
            return asw_main_window_panel;
        }
        LOG_ERROR(main, "asw_main_window_panel is nullptr");
        return nullptr;
    }
    auto* const get_composite_panel() const
    {
        if (composite_panel != nullptr)
        {
            return composite_panel;
        }
        LOG_ERROR(main, "composite_panel is nullptr");
        return nullptr;
    }
    auto* const get_image_rendering_panel() const
    {
        if (image_rendering_panel != nullptr)
        {
            return image_rendering_panel;
        }
        LOG_ERROR(main, "image_rendering_panel is nullptr");
        return nullptr;
    }
    auto* const get_import_panel() const
    {
        if (import_panel != nullptr)
        {
            return import_panel;
        }
        LOG_ERROR(main, "import_panel is nullptr");
        return nullptr;
    }
    auto* const get_info_panel() const
    {
        if (info_panel != nullptr)
        {
            return info_panel;
        }
        LOG_ERROR(main, "info_panel is nullptr");
        return nullptr;
    }
    auto* const get_view_panel() const
    {
        if (view_panel != nullptr)
        {
            return view_panel;
        }
        LOG_ERROR(main, "view_panel is nullptr");
        return nullptr;
    }
};
} // namespace holovibes
