#include <boost/algorithm/string.hpp>
#include <boost/program_options/parsers.hpp>

#include "options_parser.hh"

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
        "compute_settings,c",
        po::value<std::string>(),
        ".json config file path"
    )
    (
        "noskip_acc",
        po::bool_switch()->default_value(false),
        "Don't skip img acc frames at the beginning (default = false)"
    )
    (
        "fps,f",
        po::value<unsigned int>(),
        "Input file fps (default = infinite)"
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
        "start_frame,s",
        po::value<unsigned int>(),
        "Start frame (default = 1). Everything strictly before start frame is not read."
    )
    (
        "end_frame,e",
        po::value<unsigned int>(),
        "End frame (default = eof). Everything strictly after end frame is not read."
    )
    ;
    // clang-format on

    opts_desc_.add(general_opts_desc).add(run_opts_desc);
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
        if (vm_.count("compute_settings"))
            options_.compute_settings_path = boost::any_cast<std::string>(vm_["compute_settings"].value());
        if (vm_.count("fps"))
        {
            int fps = boost::any_cast<uint>(vm_["fps"].value());
            if (fps >= 0)
                options_.fps = fps; // Implicit cast to uint
            else
            {
                LOG_ERROR("fps value should be positive");
                exit(22);
            }
        }
        if (vm_.count("n_rec"))
        {
            int n_rec = boost::any_cast<uint>(vm_["n_rec"].value());
            if (n_rec > 0)
                options_.n_rec = n_rec; // Implicit cast to uint
            else
            {
                LOG_ERROR("n_rec value should be positive");
                exit(23);
            }
        }
        if (vm_.count("start_frame"))
        {
            int start_frame = boost::any_cast<uint>(vm_["start_frame"].value());
            if (start_frame > 0)
                options_.start_frame = start_frame; // Implicit cast to uint
            else
            {
                LOG_ERROR("start_frame value should be positive");
                exit(24);
            }
        }
        if (vm_.count("end_frame"))
        {
            int end_frame = boost::any_cast<uint>(vm_["end_frame"].value());
            if (end_frame > 0)
                options_.end_frame = end_frame; // Implicit cast to uint
            else
            {
                LOG_ERROR("end_frame value should be positive");
                exit(25);
            }
        }
        options_.record_raw = vm_["raw"].as<bool>();
        options_.verbose = vm_["verbose"].as<bool>();
        options_.noskip_acc = vm_["noskip_acc"].as<bool>();
        options_.gpu = vm_["gpu"].as<bool>();
    }
    catch (std::exception& e)
    {
        LOG_INFO("Error when parsing options: {}", e.what());
        std::exit(20);
    }

    return options_;
}

po::options_description OptionsParser::get_opts_desc() const { return opts_desc_; }
} // namespace holovibes
