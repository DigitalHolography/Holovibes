#pragma once
// without namespace
#include "tools.hh"
#include "json.hh"
using json = ::nlohmann::json;

#include "enum_record_mode.hh"

// namespace camera
#include "camera_exception.hh"

// namespace holovibes
#include "holovibes.hh"
#include "custom_exception.hh"

// namespace gui
#include "HoloWindow.hh"
#include "SliceWindow.hh"
#include "PlotWindow.hh"
#include "Filter2DWindow.hh"
#include "ui_mainwindow.h"

namespace holovibes
{
class UserInterfaceDescriptor
{

  public:
    UserInterfaceDescriptor()
    {
        std::filesystem::path holovibes_documents_path = get_user_documents_path() / "Holovibes";
        std::filesystem::create_directory(holovibes_documents_path);
        record_output_directory_ = holovibes_documents_path.string();
    }

    enum ImportType
    {
        None,
        Camera,
        File,
    };

    Holovibes& holovibes_ = Holovibes::instance();

    camera::FrameDescriptor file_fd_;

    std::unique_ptr<::holovibes::gui::RawWindow> mainDisplay = nullptr;
    std::unique_ptr<::holovibes::gui::SliceWindow> sliceXZ = nullptr;
    std::unique_ptr<::holovibes::gui::SliceWindow> sliceYZ = nullptr;
    std::unique_ptr<::holovibes::gui::RawWindow> lens_window = nullptr;
    std::unique_ptr<::holovibes::gui::RawWindow> raw_window = nullptr;
    std::unique_ptr<::holovibes::gui::Filter2DWindow> filter2d_window = nullptr;
    std::unique_ptr<::holovibes::gui::PlotWindow> plot_window_ = nullptr;

    uint window_max_size = 768;
    uint time_transformation_cuts_window_max_size = 512;
    uint auxiliary_window_max_size = 512;

    float displayAngle = 0.f;
    float xzAngle = 0.f;
    float yzAngle = 0.f;

    int displayFlip = 0;
    int xzFlip = 0;
    int yzFlip = 0;

    bool is_enabled_camera_ = false;
    double z_step_ = 0.005f;

    bool is_recording_ = false;
    unsigned record_frame_step_ = 512;
    RecordMode record_mode_ = RecordMode::RAW;

    std::string default_output_filename_{"capture"};
    std::string record_output_directory_;
    std::string file_input_directory_{"C:\\"};
    std::string batch_input_directory_{"C:\\"};

    CameraKind kCamera = CameraKind::NONE;
    ImportType import_type_ = ImportType::None;
    QString last_img_type_ = "Magnitude";

    size_t auto_scale_point_threshold_ = 100;
    ushort theme_index_ = 0;
};
} // namespace holovibes