/*! \file main.cc
 *
 * \brief Starts the application in CLI mode or GUI mode (light ui mode if previously
 * closed in light ui mode) in function of passed parameters.
 *
 * This file also check if a GPU in installed and if the CUDA version is greater than 3.5.
 * On each run in release mode, data from the local AppData (preset, camera ini, shaders, ...)
 * are copied to the user AppData.
 */

#include <QApplication>
#include <QLocale>
#include <QPixmap>
#include <QSplashScreen>

#include "API.hh"
#include "options_parser.hh"
#include "MainWindow.hh"
#include "frame_desc.hh"
#include "holovibes_config.hh"
#include "logger.hh"

#include "cli.hh"

#include <spdlog/spdlog.h>

#define MIN_CUDA_VERSION 35

static void check_cuda_graphic_card(bool gui)
{
    std::string error_message;
    int nDevices;

    /* Checking for Compute Capability */
    if (cudaGetDeviceCount(&nDevices) == cudaSuccess)
    {
        cudaDeviceProp props;
        int device;

        cudaGetDevice(&device);
        cudaGetDeviceProperties(&props, device);

        // Check cuda version
        if (props.major * 10 + props.minor >= MIN_CUDA_VERSION)
            return;
        else
            error_message = "CUDA graphic card not supported.\n";
    }
    else
        error_message = "No CUDA graphic card detected.\n"
                        "You will not be able to run Holovibes.\n\n"
                        "Try to update your graphic drivers.";

    if (gui)
    {
        QMessageBox messageBox;
        messageBox.critical(0, "No CUDA graphic card detected", QString::fromUtf8(error_message.c_str()));
        messageBox.setFixedSize(800, 300);
    }
    else
        LOG_CRITICAL("{}", error_message);

    std::exit(11);
}

static int start_gui(holovibes::Holovibes& holovibes, int argc, char** argv, const std::string filename = "")
{
    holovibes.is_cli = false;
    LOG_TRACE(" ");

    QLocale::setDefault(QLocale("en_US"));
    // Create the Qt app
    QApplication app(argc, argv);
    app.setWindowIcon(QIcon(":/assets/icons/Holovibes.ico"));

    LOG_TRACE(" ");
    check_cuda_graphic_card(true);
    QSplashScreen splash(QPixmap(":/assets/icons/holovibes_logo.png"));
    splash.show();

    LOG_TRACE(" ");

    // Hide the possibility to close the console while using Holovibes
    HWND hwnd = GetConsoleWindow();
    HMENU hmenu = GetSystemMenu(hwnd, FALSE);
    EnableMenuItem(hmenu, SC_CLOSE, MF_GRAYED);
    LOG_TRACE(" ");

    // Create the window object that inherit from QMainWindow
    holovibes::gui::MainWindow window;

    LOG_TRACE(" ");
    if (holovibes::api::is_light_ui_mode())
        window.light_ui_->show();
    else
        window.show();
    LOG_TRACE(" ");
    splash.finish(&window);

    if (!filename.empty())
    {
        window.start_import(QString(filename.c_str()));
        LOG_INFO("Imported file {}", filename.c_str());
    }
    LOG_TRACE();

    // Resizing horizontally the window before starting
    window.layout_toggled();
    // Launch the Qt app
    return app.exec();
}

static void print_version() { std::cout << "Holovibes " << __HOLOVIBES_VERSION__ << std::endl; }

static void print_help(holovibes::OptionsParser parser)
{
    print_version();
    std::cout << "Usage: ./Holovibes.exe [OPTIONS]" << std::endl;
    std::cout << parser.get_opts_desc();
}

// Copy all files from src path to dest path (the directories will be created if not exist)
static void copy_files(const std::filesystem::path src, std::filesystem::path dest)
{
    std::filesystem::create_directories(dest);

    for (const auto& entry : std::filesystem::directory_iterator(src))
    {
        std::filesystem::path file = entry.path();
        std::filesystem::path dest_file = dest / file.filename();
        if (!std::filesystem::exists(dest_file))
            std::filesystem::copy(file, dest_file);
    }
}

int main(int argc, char* argv[])
{
    {
        std::unique_lock lock(holovibes::Logger::map_mutex_);
        holovibes::Logger::add_thread(std::this_thread::get_id(), ":main");
    }

    LOG_INFO("Start Holovibes");
    LOG_TRACE("hello");

    holovibes::OptionsParser parser;
    holovibes::OptionsDescriptor opts = parser.parse(argc, argv);

    if (opts.print_help)
    {
        print_help(parser);
        std::exit(0);
    }

    if (opts.print_version)
    {
        print_version();
        std::exit(0);
    }

    if (opts.benchmark)
        holovibes::api::set_benchmark_mode(true);

    holovibes::Holovibes& holovibes = holovibes::Holovibes::instance();

    int ret = 0;
    try
    {

#ifdef NDEBUG
        /*
            If we are on release mode, at first boot copy the reference files from the local AppData to the real user
            AppData/Roaming/Holovibes location.
            We use GET_EXE_DIR completed with macros instead of absolute paths to avoid crashing during
            debugging.
            It may be cleaner to propagate files during instalation (for release mode) and during compilation
            (for debug mode) but hard to do...
        */
        copy_files(RELATIVE_PATH(__CAMERAS_CONFIG_REFERENCE__), RELATIVE_PATH(__CAMERAS_CONFIG_FOLDER_PATH__));
        copy_files(RELATIVE_PATH(__PRESET_REFERENCE__), RELATIVE_PATH(__PRESET_FOLDER_PATH__));
        copy_files(RELATIVE_PATH(__CONVOLUTION_KERNEL_REFERENCE__), RELATIVE_PATH(__CONVOLUTION_KERNEL_FOLDER_PATH__));
        copy_files(RELATIVE_PATH(__INPUT_FILTER_REFERENCE__), RELATIVE_PATH(__INPUT_FILTER_FOLDER_PATH__));
        copy_files(RELATIVE_PATH(__SHADER_REFERENCE__), RELATIVE_PATH(__SHADER_FOLDER_PATH__));

#endif

        if (opts.input_path && opts.output_path)
        {
            check_cuda_graphic_card(false);
            ret = cli::start_cli(holovibes, opts);
        }
        else if (opts.input_path)
            ret = start_gui(holovibes, argc, argv, opts.input_path.value());
        else
            ret = start_gui(holovibes, argc, argv);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Uncaught exception: {}", e.what());
        ret = 10;
    }

    spdlog::shutdown();

    return ret;
}
