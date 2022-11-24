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

int main(int argc, char* argv[])
{
    LOG_INFO(main, "Start Holovibes");

    holovibes::OptionsParser parser;
    holovibes::OptionsDescriptor opts = parser.parse(argc, argv);

    if (opts.print_help)
    {
        holovibes::api::print_help(parser);
        std::exit(0);
    }

    if (opts.print_version)
    {
        holovibes::api::print_version();
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
        LOG_ERROR(main, "Uncaught exception: {}", e.what());
        return 1;
    }

    return 0;
}
