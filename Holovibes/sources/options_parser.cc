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
    (
        "benchmark,b",
        po::bool_switch()->default_value(false),
        "Benchmark mode (default = false)"
    )
    (
        "frame_skip",
        po::value<unsigned int>(),
        "Frame to skip between each frame recorded"
    )
    (
        "mp4_fps",
        po::value<unsigned int>(),
        "MP4 fps value, default 24. Warning : it may crash for big values"
    )
    (
        "moments_record",
        po::bool_switch()->default_value(false),
        "Record moments (default = false)"
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
        po::store(po::command_line_parser(argc, argv).options(opts_desc_).run(), vm_);
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
        options_.benchmark = vm_["benchmark"].as<bool>();
        if (vm_.count("frame_skip"))
        {
            int frame_skip = boost::any_cast<uint>(vm_["frame_skip"].value());
            if (frame_skip >= 0)
                options_.frame_skip = frame_skip; // Implicit cast to uint
            else
            {
                LOG_ERROR("frame_skip value should be positive");
                exit(28);
            }
        }
        if (vm_.count("mp4_fps"))
        {
            int mp4_fps = boost::any_cast<uint>(vm_["mp4_fps"].value());
            if (mp4_fps > 0)
                options_.mp4_fps = mp4_fps; // Implicit cast to uint
            else
            {
                LOG_ERROR("mp4_fps value should be positive");
                exit(31);
            }
        }
        options_.moments_record = vm_["moments_record"].as<bool>();
    }
    catch (std::exception& e)
    {
        LOG_INFO("Error when parsing options: {}", e.what());
        std::exit(20);
    }

    // Both catch blocks below are never reached as the base class std::exception will catch them before...
    // Left them as dead code for potential future use as fixing them (by moving them above the std::exception catch) would require to change some of the test suite expected return codes
    #if 0
    catch (const po::invalid_option_value& ex)
    {
        // Gérer le cas où une option reçoit une valeur invalide
        LOG_ERROR("Invalid option value: {}", ex.what());
        std::exit(27); // Utiliser un code d'erreur spécifique
    }
    catch (const po::unknown_option& ex)
    {
        // Gérer le cas où un flag inconnu est passé
        LOG_ERROR("Unknown option: {}", ex.what());
        std::exit(26); // Utiliser un code d'erreur spécifique
    }
    #endif

    return options_;
}

po::options_description OptionsParser::get_opts_desc() const { return opts_desc_; }
} // namespace holovibes
