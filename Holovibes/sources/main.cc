/*! \mainpage Holovibes

    Documentation for developpers. \n
*/

#include <QApplication>
#include <QLocale>
#include <QPixmap>
#include <QSplashScreen>

#include "options_parser.hh"
#include "MainWindow.hh"
#include "frame_desc.hh"
#include "compute_descriptor.hh"
#include "logger.hh"
#include "cli.hh"

static void check_cuda_graphic_card(bool gui)
{
    std::string error_message;
    int device;
    int nDevices;
    int min_compute_capability = 35;
    int max_compute_capability = 86;
    int compute_capability;
    cudaError_t status;
    cudaDeviceProp props;

    /* Checking for Compute Capability */
    if ((status = cudaGetDeviceCount(&nDevices)) == cudaSuccess)
    {
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&props, device);

        compute_capability = props.major * 10 + props.minor;

        if (compute_capability >= min_compute_capability && compute_capability <= max_compute_capability)
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
    {
        LOG_WARN << error_message;
    }
    std::exit(1);
}

static int start_gui(holovibes::Holovibes& holovibes, int argc, char** argv, const std::string filename = "")
{
    QLocale::setDefault(QLocale("en_US"));
    // Create the Qt app
    QApplication app(argc, argv);

    check_cuda_graphic_card(true);
    QSplashScreen splash(QPixmap("holovibes_logo.png"));
    splash.show();

    // Hide the possibility to close the console while using Holovibes
    HWND hwnd = GetConsoleWindow();
    HMENU hmenu = GetSystemMenu(hwnd, FALSE);
    EnableMenuItem(hmenu, SC_CLOSE, MF_GRAYED);

    // Create the window object that inherit from QMainWindow
    holovibes::gui::MainWindow window(holovibes);
    window.show();
    splash.finish(&window);
    holovibes.get_cd().register_observer(window);

    // Resizing horizontally the window before starting
    window.layout_toggled();

    if (filename != "")
    {
        window.import_file(QString(filename.c_str()));
        window.import_start();
    }

    // Launch the Qt app
    return app.exec();
}

static void print_version() { std::cerr << "Holovibes " << __HOLOVIBES_VERSION__; }

static void print_help(holovibes::OptionsParser parser)
{
    print_version();
    std::cerr << std::endl << "Usage: ./Holovibes.exe [OPTIONS]" << std::endl;
    std::cerr << parser.get_opts_desc();
}

int main(int argc, char* argv[])
{

#ifndef _DEBUG
    // Put every log message in "everything.log":
    loguru::add_file("everything.log", loguru::Append, loguru::Verbosity_MAX);

    // Only log INFO, WARNING, ERROR and FATAL to "latest_readable.log":
    loguru::add_file("latest_readable.log", loguru::Truncate, loguru::Verbosity_INFO);

    loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
#else
    loguru::g_stderr_verbosity = 2;
#endif

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

    holovibes::Holovibes& holovibes = holovibes::Holovibes::instance();

    int ret = 0;
    try
    {
        if (opts.input_path && opts.output_path)
        {
            check_cuda_graphic_card(false);
            ret = cli::start_cli(holovibes, opts);
        }
        else if (opts.input_path)
        {
            ret = start_gui(holovibes, argc, argv, opts.input_path.value());
        }
        else
        {
            ret = start_gui(holovibes, argc, argv);
        }
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << "Uncaught exception: " << e.what();
        ret = 1;
    }

    return ret;
}
