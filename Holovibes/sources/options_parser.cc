/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include <boost/algorithm/string.hpp>
#include <boost/program_options/parsers.hpp>

#include "options_parser.hh"
#include "config.hh"

namespace holovibes
{
OptionsParser::OptionsParser()
    : vm_()
{
    po::options_description general_opts_desc("General");
    general_opts_desc.add_options()("version,v",
                                    "Print the version number and exit.")(
        "help,h",
        "Print a summary of CLI options and exit.");

    po::options_description run_opts_desc("Run");
    run_opts_desc.add_options()("input,i",
                                po::value<std::string>(),
                                "Import a .holo file.")(
        "output,o",
        po::value<std::string>(),
        "Export a .holo file.")("input-fps",
                                po::value<unsigned int>(),
                                "Set holo file input FPS.")(
        "output-nb-frames",
        po::value<unsigned int>(),
        "Set number of frames for the output file.")(
        "record-raw",
        po::bool_switch()->default_value(false),
        "Set flag to record raw (false by default)");

    opts_desc_.add(general_opts_desc).add(run_opts_desc);
}

OptionsDescriptor OptionsParser::parse(int argc, char* const argv[])
{
    try
    {
        // Parse options
        po::store(po::command_line_parser(argc, argv)
                      .options(opts_desc_)
                      .allow_unregistered()
                      .run(),
                  vm_);
        po::notify(vm_);

        // Handle general options
        options_.print_help = vm_.count("help");
        options_.print_version = vm_.count("version");

        if (vm_.count("input"))
            options_.input_path =
                boost::any_cast<std::string>(vm_["input"].value());
        if (vm_.count("output"))
            options_.output_path =
                boost::any_cast<std::string>(vm_["output"].value());
        if (vm_.count("input-fps"))
            options_.input_fps =
                boost::any_cast<unsigned int>(vm_["input-fps"].value());
        if (vm_.count("output-nb-frames"))
            options_.output_nb_frames =
                boost::any_cast<unsigned int>(vm_["output-nb-frames"].value());
        options_.record_raw = vm_["record-raw"].as<bool>();
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
        std::exit(1);
    }

    return options_;
}

po::options_description OptionsParser::get_opts_desc() const
{
    return opts_desc_;
}
} // namespace holovibes