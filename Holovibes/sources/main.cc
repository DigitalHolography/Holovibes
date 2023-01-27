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
#include "global_state_holder.hh"
#include "API.hh"

#include "cli.hh"
#include "gui.hh"

using namespace holovibes;

inline void print_version() { std::cout << "Holovibes " << __HOLOVIBES_VERSION__ << std::endl; }

inline void print_help(OptionsParser& parser)
{
    print_version();
    std::cout << "Usage: ./Holovibes.exe [OPTIONS]" << std::endl;
    std::cout << parser.get_opts_desc();
}

int main(int argc, char* argv[])
{
    holovibes::Logger::add_thread(std::this_thread::get_id(), ":main");

    LOG_INFO("Start Holovibes");
    LOG_INFO("Using config folder: \"{}\"", __CONFIG_FOLDER__.string());

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

    try
    {
        if (opts.input_path && opts.output_path)
        {
            holovibes::cli::start_cli(opts);
        }
        else if (opts.input_path)
        {
            holovibes::gui::start_gui(argc, argv, opts.input_path.value());
        }
        else
        {
            holovibes::gui::start_gui(argc, argv);
        }
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Uncaught exception: {}", e.what());
        return 1;
    }

    return 0;
}
