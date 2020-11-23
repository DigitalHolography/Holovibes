/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

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
		general_opts_desc.add_options()
			("version,v", "Print the version number and exit.")
			("help,h", "Print a summary of CLI options and exit.");

		po::options_description run_opts_desc("Run");
		run_opts_desc.add_options()
			("input,i", po::value<std::string>(), "Import a .holo file.")
			("output,o", po::value<std::string>(), "Export a .holo file.")
			("input-fps", po::value<unsigned int>(), "Set holo file input FPS.")
			("output-nb-frames", po::value<unsigned int>(), "Set number of frames for the output file.")
			("record-raw", po::bool_switch()->default_value(false), "Set flag to record raw (false by default)");

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
				.run(), vm_);
			po::notify(vm_);

			// Handle general options
			options_.print_help = vm_.count("help");
			options_.print_version = vm_.count("version");

			if (vm_.count("input"))
				options_.input_path = boost::any_cast<std::string>(vm_["input"].value());
			if (vm_.count("output"))
				options_.output_path = boost::any_cast<std::string>(vm_["output"].value());
			if (vm_.count("input-fps"))
				options_.input_fps = boost::any_cast<unsigned int>(vm_["input-fps"].value());
			if (vm_.count("output-nb-frames"))
				options_.output_nb_frames = boost::any_cast<unsigned int>(vm_["output-nb-frames"].value());
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
}