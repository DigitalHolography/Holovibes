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
#include "input_frame_file_factory.hh"
#include "logger.hh"

#include "frame_record_worker.hh"

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

static void start_cli(holovibes::Holovibes& holovibes,
                      const holovibes::OptionsDescriptor& opts)
{
    check_cuda_graphic_card(false);
    holovibes.start_information_display(true);

    std::string input_path = opts.input_path.value();

    holovibes::io_files::InputFrameFile* input_frame_file = nullptr;

    try
    {
        input_frame_file =
            holovibes::io_files::InputFrameFileFactory::open(input_path);
    }
    catch (const holovibes::io_files::FileException& e)
    {
        LOG_ERROR(e.what());
        return;
    }

    const camera::FrameDescriptor& fd =
        input_frame_file->get_frame_descriptor();
    size_t input_nb_frames = input_frame_file->get_total_nb_frames();

    const unsigned int input_fps = opts.input_fps.value_or(60);
    holovibes.init_input_queue(fd);
    holovibes.start_file_frame_read(input_path,
                                    true,
                                    input_fps,
                                    0,
                                    input_nb_frames,
                                    false);

    input_frame_file->import_compute_settings(holovibes.get_cd());

    holovibes.update_cd_for_cli(input_fps);
    holovibes.start_compute();

    // Start recording.
    holovibes::worker::FrameRecordWorker frame_record_worker(
        opts.output_path.value(),
        opts.output_nb_frames.value_or(input_nb_frames),
        opts.record_raw,
        false);
    frame_record_worker.run();

    holovibes.stop_all_worker_controller();

    delete input_frame_file;
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

    if (opts.input_path)
    {
        if (opts.output_path)
            start_cli(holovibes, opts);
        else // start gui
            start_gui(holovibes, argc, argv, opts.input_path.value());

        return 0;
    }

    return start_gui(holovibes, argc, argv);
}
