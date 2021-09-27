#include <boost/algorithm/string.hpp>
#include <boost/program_options/parsers.hpp>

#include "options_parser.hh"
#include "config.hh"
#include "logger.hh"

namespace holovibes
{
OptionsParser::OptionsParser()
    : vm_()
{
    // clang-format off
    po::options_description general_opts_desc("General");
    general_opts_desc.add_options()
    (
        "version",
        "Print the version number and exit"
    )
    (
        "help,h",
        "Print a summary of CLI options and exit"
    );

    po::options_description run_opts_desc("Run");
    run_opts_desc.add_options()
    (
        "verbose,v",
        po::bool_switch()->default_value(false),
        "Verbose mode (default = false)"
    )
    (
        "input,i",
        po::value<std::string>(),
        "Input file path"
    )
    (
        "output,o",
        po::value<std::string>(),
        "Output file path"
    )
    (
        "ini",
        po::value<std::string>(),
        ".ini config file path (default = holovibes.ini)"
    )
    (
        "convolution,c",
        po::value<std::string>(),
        "Convolution matrix path (default = no convolution)"
    )
    (
        "divide,d",
        po::bool_switch()->default_value(false),
        "Divide by convolution matrix (default = false)"
    )
    (
        "noskip_acc",
        po::bool_switch()->default_value(false),
        "Don't skip img acc frames at the beginning (default = false)"
    )
    (
        "fps,f",
        po::value<unsigned int>(),
        "Input file fps (default = 60)"
    )
    (
        "n_rec,n",
        po::value<unsigned int>(),
        "Number of frames to record (default = same as input file)"
    )
    (
        "raw",
        po::bool_switch()->default_value(false),
        "Enable raw recording (default = false)"
    )
    (
        "gpu",
        po::bool_switch()->default_value(false),
        "Load file in GPU (default = false)"
    )
    (
        "start_index,s",
        po::value<unsigned int>(),
        "Start index (default = 1)"
    )
    (
        "end_index,e",
        po::value<unsigned int>(),
        "End index (default = eof)"
    )
        // clang-format on

        opts_desc_.add(general_opts_desc)
        .add(run_opts_desc);
}

OptionsDescriptor OptionsParser::parse(int argc, char* const argv[])
{
    try
    {
        // Parse options
        po::store(po::command_line_parser(argc, argv).options(opts_desc_).allow_unregistered().run(), vm_);
        po::notify(vm_);

        // Handle general options
        options_.print_help = vm_.count("help");
        options_.print_version = vm_.count("version");

        if (vm_.count("input"))
            options_.input_path = boost::any_cast<std::string>(vm_["input"].value());
        if (vm_.count("output"))
            options_.output_path = boost::any_cast<std::string>(vm_["output"].value());
        if (vm_.count("ini"))
            options_.ini_path = boost::any_cast<std::string>(vm_["ini"].value());
        if (vm_.count("convolution"))
            options_.convo_path = boost::any_cast<std::string>(vm_["convolution"].value());
        if (vm_.count("fps"))
            options_.fps = boost::any_cast<unsigned int>(vm_["fps"].value());
        if (vm_.count("n_rec"))
            options_.n_rec = boost::any_cast<unsigned int>(vm_["n_rec"].value());
        if (vm_.count("start_index"))
            options_.n_rec = boost::any_cast<unsigned int>(vm_["start_index"].value());
        if (vm_.count("end_index"))
            options_.n_rec = boost::any_cast<unsigned int>(vm_["end_index"].value());
        options_.record_raw = vm_["raw"].as<bool>();
        options_.verbose = vm_["verbose"].as<bool>();
        options_.divide_convo = vm_["divide"].as<bool>();
        options_.noskip_acc = vm_["noskip_acc"].as<bool>();
        options_.gpu = vm_["gpu"].as<bool>();
    }
    catch (std::exception& e)
    {
        LOG_INFO << e.what() << std::endl;
        std::exit(1);
    }

    return options_;
}

po::options_description OptionsParser::get_opts_desc() const { return opts_desc_; }
} // namespace holovibes
