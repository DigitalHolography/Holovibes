#pragma once

#define UID_CONVOLUTION_TYPE_DEFAULT "None"
#define UID_FILTER_TYPE_DEFAULT "None"
#define UID_PRESET_TYPE_DEFAULT "None"

// without namespace
#include "tools.hh"

#include "enum_record_mode.hh"
#include "enum_import_type.hh"

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

namespace holovibes
{

class UserInterfaceDescriptor
{
  private:
    UserInterfaceDescriptor()
    {
        std::filesystem::path holovibes_documents_path = get_user_documents_path() / "Holovibes";
        std::filesystem::create_directory(holovibes_documents_path);
        record_output_directory_ = holovibes_documents_path.string();
    }

  public:
    static UserInterfaceDescriptor& instance()
    {
        static UserInterfaceDescriptor instance{};
        return instance;
    }

    std::unique_ptr<::holovibes::gui::RawWindow> mainDisplay = nullptr;
    std::unique_ptr<::holovibes::gui::SliceWindow> sliceXZ = nullptr;
    std::unique_ptr<::holovibes::gui::SliceWindow> sliceYZ = nullptr;
    std::unique_ptr<::holovibes::gui::RawWindow> lens_window = nullptr;
    std::unique_ptr<::holovibes::gui::RawWindow> raw_window = nullptr;
    std::unique_ptr<::holovibes::gui::Filter2DWindow> filter2d_window = nullptr;
    std::unique_ptr<::holovibes::gui::PlotWindow> plot_window_ = nullptr;
    std::unique_ptr<::holovibes::gui::AdvancedSettingsWindow> advanced_settings_window_ = nullptr;

    size_t auto_scale_point_threshold_ = 100;

    std::string output_filename_{"capture"};
    std::string record_output_directory_;
    std::string file_input_directory_{"C:\\"};
};
} // namespace holovibes
