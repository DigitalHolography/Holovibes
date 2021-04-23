/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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
    int nDevices;
    cudaError_t status = cudaGetDeviceCount(&nDevices);
    if (status == cudaSuccess)
        return;

    std::string error_message = "No CUDA graphic card detected.\n"
                                "You will not be able to run Holovibes.\n\n"
                                "Try to update your graphic drivers.";

    if (gui)
    {
        QMessageBox messageBox;
        messageBox.critical(0,
                            "No CUDA graphic card detected",
                            QString::fromUtf8(error_message.c_str()));
        messageBox.setFixedSize(800, 300);
    }
    else
    {
        LOG_WARN(error_message);
    }
    std::exit(1);
}

static int start_gui(holovibes::Holovibes& holovibes,
                     int argc,
                     char** argv,
                     const std::string filename = "")
{
    QLocale::setDefault(QLocale("en_US"));
    QApplication app(argc, argv);
    check_cuda_graphic_card(true);
    QSplashScreen splash(QPixmap("holovibes_logo.png"));
    splash.show();

    // Hide the possibility to close the console while using Holovibes
    HWND hwnd = GetConsoleWindow();
    HMENU hmenu = GetSystemMenu(hwnd, FALSE);
    EnableMenuItem(hmenu, SC_CLOSE, MF_GRAYED);

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

    return app.exec();
}

static void print_version()
{
    std::cout << "Holovibes " << holovibes::version << std::endl;
}

static void print_help(holovibes::OptionsParser parser)
{
    print_version();
    std::cout << std::endl << "Usage: ./Holovibes.exe [OPTIONS]" << std::endl;
    std::cout << parser.get_opts_desc();
}

int main(int argc, char* argv[])
{
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
        std::cerr << "Uncaught exception: " << e.what() << std::endl;
    }
    return ret;
}
