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
#include "options_descriptor.hh"

namespace holovibes
{
	OptionsParser::OptionsParser(OptionsDescriptor& opts)
		: opts_(opts)
		, pos_desc_()
		, general_opts_desc_("General")
		, import_opts_desc_("Import")
		, features_opts_desc_("Features")
		, compute_opts_desc_("Computation options")
		, merge_opts_desc_()
		, vm_()
	{
		init_general_options();
		init_compute_options();
	}

	void OptionsParser::init_general_options()
	{
		general_opts_desc_.add_options()
			("version", "Print the version number of Holovibes and exit.")
			("help,h", "Print a summary of the command-line options to Holovibes and exit.")
			("nogui", "Disable graphical user interface.")
			("import,i", "Enable import file mode (Overide a camera). Requires f, img-dim parameters.");
	}

	void OptionsParser::init_features_options(const bool is_no_gui, const bool is_import_mode_enable)
	{
		features_opts_desc_.add_options()
			("queuesize,q",
			po::value<int>()->default_value(default_queue_size),
			"Size of both queues arg in number of images");

		if (is_import_mode_enable)
		{
			import_opts_desc_.add_options()
				("file,f",
				po::value<std::string>()
				->required(),
				"File path used by import mode. (Requierd in import mode).")

				("img-dim",
				po::value<std::vector<unsigned int>>()
				->multitoken()
				->required(),
				"Set sizes of imported file (Requierd in import mode)."
				"The first argument gives the with. "
				"The second argument gives the height.")

				("fps",
				po::value<unsigned int>()
				->default_value(default_fps),
				"Set frame imported per seconds. (Useless with nogui arg)")

				("depth",
				po::value<unsigned int>()
				->default_value(default_depth),
				"Set depth of imported file.")

				("big-endian",
				po::value<bool>()
				->default_value(default_is_big_endian),
				"Set big endian mode of imported file.")

				("frame_start",
				po::value<unsigned int>()
				->default_value(1),
				"Set first frame id imported.")

				("frame_end",
				po::value<unsigned int>()
				->default_value(65536),
				"Set the max frame id imported.")
				;
		}

		if (is_no_gui)
		{
			features_opts_desc_.add_options()
				("write,w",
				po::value<std::vector<std::string>>()
				->multitoken()
				->required(),
				"Record a sequence of images in the given path. "
				"The first argument gives the number of images to record. "
				"The second argument gives the filepath where frames will be recorded.")

				("float-output,o",
				"Set on float output, Requires compute mode.")
				;
			if (!is_import_mode_enable)
			{
				features_opts_desc_.add_options()
					("cameramodel,c",
					po::value<std::string>()
					->required(),
					"Set the camera to use: pike/xiq/ids/pixelfly/ixon/edge.")
					;
			}
		}
		else
		{
			features_opts_desc_.add_options()
				("display,d",
				po::value<std::vector<int>>()
				->multitoken(),
				"Set default sizes of realtime display."
				"The first argument gives the square size of the display. "
				"The second optional argument specify the height.")
				;
		}
	}

	void OptionsParser::init_compute_options()
	{
		compute_opts_desc_.add_options()
			("1fft",
			"Enable the 1-FFT method: Fresnel transform. Requires n, p, l, z parameters.")

			("2fft",
			"Enable the 2-FFT method: Angular spectrum propagation approache. Requires n, p, l, z parameters.")

			("nSize,n",
			po::value<unsigned short>(),
			"Number of samples N.")

			("pindex,p",
			po::value<unsigned short>(),
			"Select the p-th component of the DFT, p must be defined in {0, ..., N - 1}.")

			("lambda,l",
			po::value<float>(),
			"Light wavelength.")

			("zdistance,z",
			po::value<float>(),
			"The parameter z corresponds to the sensor-to-object distance.")

			("viewmode",
			po::value<std::string>(),
			"Select the view mode: magnitude/squaredmagnitude/argument.")

			("log",
			"Apply log10 on output frames.")

			("nofftshift",
			"Disable FFT shifting.")

			("contrastmin",
			po::value<float>(),
			"Enable contrast and set min value."
			"Argument use logarithmic scale.")

			("contrastmax",
			po::value<float>(),
			"Enable contrast and set max value."
			"Argument use logarithmic scale.")
			;
	}

	void OptionsParser::init_merge_options()
	{
		merge_opts_desc_.add(general_opts_desc_);
		merge_opts_desc_.add(import_opts_desc_);
		merge_opts_desc_.add(features_opts_desc_);
		merge_opts_desc_.add(compute_opts_desc_);
	}

	void OptionsParser::parse_general_options(int argc, char* const argv[])
	{
		/* First parsing to check help/version options. */
		po::store(po::command_line_parser(argc, argv)
			.options(general_opts_desc_)
			.allow_unregistered()
			.run(), vm_);
		po::notify(vm_);
	}

	bool OptionsParser::get_is_gui_enabled()
	{
		return !vm_.count("nogui");
	}

	bool OptionsParser::get_is_import_mode()
	{
		return vm_.count("import");
	}

	void OptionsParser::parse_features_compute_options(int argc, char* const argv[])
	{
		po::store(
			po::command_line_parser(argc, argv)
			.options(merge_opts_desc_)
			.positional(pos_desc_)
			.run(), vm_);
		po::notify(vm_);
	}

	void OptionsParser::parse(int argc, char* const argv[])
	{
		bool succeed = false;

		try
		{
			parse_general_options(argc, argv);

			opts_.is_gui_enabled = get_is_gui_enabled();
			opts_.is_import_mode_enabled = get_is_import_mode();

			init_features_options(!opts_.is_gui_enabled, opts_.is_import_mode_enabled);
			init_merge_options();

			/* May exit here. */
			proceed_help();

			parse_features_compute_options(argc, argv);

			proceed_features();
			proceed_compute();

			if (!opts_.is_gui_enabled && opts_.is_compute_enabled)
				check_compute_params();

			succeed = true;
		}
		catch (po::unknown_option& e)
		{
			std::cerr << "[CLI] " << e.what() << std::endl;
		}
		catch (po::invalid_option_value &e)
		{
			std::cerr << "[CLI] " << e.what() << std::endl;
		}
		catch (po::too_many_positional_options_error& e)
		{
			std::cerr << "[CLI] " << e.what() << std::endl;
		}
		catch (po::required_option& e)
		{
			std::cerr << "[CLI] " << e.what() << std::endl;
		}
		catch (po::invalid_command_line_syntax& e)
		{
			std::cerr << "[CLI] " << e.what() << std::endl;
		}
		catch (std::runtime_error &e)
		{
			std::cerr << "[CLI] " << e.what() << std::endl;
		}

		if (!succeed)
		{
			print_help();
			std::exit(1);
		}
	}

	void OptionsParser::print_help()
	{
		print_version();
		std::cout << std::endl << "Usage: ./holovibes.exe [OPTIONS]" << std::endl
			<< "This help message depends on --nogui parameter (more options available)." << std::endl;
		std::cout << merge_opts_desc_;
	}

	void OptionsParser::print_version()
	{
		std::cout << std::endl << "Holovibes " << version << std::endl;
	}

	void OptionsParser::proceed_help()
	{
		if (vm_.count("help"))
		{
			print_help();
			std::exit(0);
		}

		if (vm_.count("version"))
		{
			print_version();
			std::exit(0);
		}
	}

	void OptionsParser::proceed_features()
	{
		if (vm_.count("cameramodel"))
		{
			const std::string& camera = vm_["cameramodel"].as<std::string>();

			if (boost::iequals(camera, "xiq"))
				opts_.camera = CameraKind::xiQ;
			else if (boost::iequals(camera, "ids"))
				opts_.camera = CameraKind::IDS;
			else if (boost::iequals(camera, "pike"))
				opts_.camera = CameraKind::Pike;
			else if (boost::iequals(camera, "pixelfly"))
				opts_.camera = CameraKind::Pixelfly;
			else if (boost::iequals(camera, "ixon"))
				opts_.camera = CameraKind::Ixon;
			else if (boost::iequals(camera, "edge"))
				opts_.camera = CameraKind::Edge;
			else
				throw std::runtime_error("unknown camera model");
		}

		if (vm_.count("display"))
		{
			const std::vector<unsigned int>& display_size =
				vm_["display"].as<std::vector<unsigned int>>();

			if (!display_size.empty())
			{
				/* Display size check. */
				if (display_size[0] < display_size_min ||
					display_size.size() >= 2 && (display_size[1] < display_size_min))
				{
					throw std::runtime_error("display width/height is too small (<100)");
				}

				opts_.gl_window_width = display_size[0];
				opts_.gl_window_height = display_size[0];

				if (display_size.size() > 1)
					opts_.gl_window_height = display_size[1];
			}
			else
			{
				/* This case should not happen. */
				assert(!"Display vector<int> is empty");
			}

			opts_.is_gl_window_enabled = true;
		}

		if (vm_.count("queuesize"))
		{
			const int queue_size = vm_["queuesize"].as<int>();

			if (queue_size > 0)
			{
				// TODO: set queue size in Config
				// Old code:
				// opts_.input_max_queue_size = queue_size;
				// opts_.output_max_queue_size = queue_size;
			}
			else
				throw std::runtime_error("queue size is too small");
		}

		if (vm_.count("write"))
		{
			const std::vector<std::string>& args =
				vm_["write"].as<std::vector<std::string>>();

			if (args.size() == 2)
			{
				try
				{
					int n_img = boost::lexical_cast<int>(args[0]);
					const std::string& filepath = args[1];
					if (filepath.empty())
						throw std::runtime_error("record filepath is empty");

					opts_.recorder_n_img = n_img;
					opts_.recorder_filepath = filepath;
				}
				catch (boost::bad_lexical_cast&)
				{
					throw std::runtime_error("wrong record first parameter (must be a number)");
				}
			}
			else
				throw std::runtime_error("-w/--write expects 2 arguments");

			opts_.is_recorder_enabled = true;
		}

		if (opts_.is_import_mode_enabled)
		{
			opts_.file_src = vm_["file"].as<std::string>();

			if (vm_["img-dim"].as<std::vector<unsigned int>>().size() != 2)
				throw std::runtime_error("img-dim request 2 integer");

			opts_.file_image_width = vm_["img-dim"].as<std::vector<unsigned int>>()[0];

			opts_.file_image_height = vm_["img-dim"].as<std::vector<unsigned int>>()[1];

			opts_.file_image_depth = vm_["depth"].as<unsigned int>();

			opts_.file_is_big_endian = vm_["big-endian"].as<bool>();

			opts_.fps = vm_["fps"].as<unsigned int>();

			opts_.spanStart = vm_["frame_start"].as<unsigned int>();

			opts_.spanEnd = vm_["frame_end"].as<unsigned int>();
			if (opts_.spanStart > opts_.spanEnd)
				throw std::runtime_error("frame_start must be smaller than frame_end");
		}
		opts_.is_float_output_enabled = vm_.count("float-output");
	}

	void OptionsParser::proceed_compute()
	{
		if (vm_.count("1fft"))
		{
			opts_.is_compute_enabled = true;
			opts_.cd.algorithm = Algorithm::FFT1;
		}

		if (vm_.count("2fft"))
		{
			if (opts_.is_compute_enabled)
				throw std::runtime_error("1fft method already selected");

			opts_.is_compute_enabled = true;
			opts_.cd.algorithm = Algorithm::FFT2;
		}

		if (vm_.count("nSize"))
		{
			const unsigned short nSize = vm_["nSize"].as<unsigned short>();
			if (nSize <= 0)
				throw std::runtime_error("--nSize parameter must be strictly positive");

			opts_.cd.nSize = nSize;

			if (opts_.cd.nSize >= global::global_config.input_queue_max_size)
				throw std::runtime_error("--nSize can not be greater than the input_max_queue_size");
		}

		if (vm_.count("pindex"))
		{
			const unsigned short pindex = vm_["pindex"].as<unsigned short>();
			if (pindex < 0 || pindex >= opts_.cd.nSize)
				throw std::runtime_error("--pindex parameter must be defined in {0, ..., nSize - 1}.");
			opts_.cd.pindex = pindex;
		}

		if (vm_.count("lambda"))
		{
			const float lambda = vm_["lambda"].as<float>();
			if (lambda <= 0.0000f)
				throw std::runtime_error("--lambda parameter must be strictly positive");
			opts_.cd.lambda = lambda;
		}

		if (vm_.count("zdistance"))
		{
			const float zdistance = vm_["zdistance"].as<float>();
			opts_.cd.zdistance = zdistance;
		}

		if (vm_.count("viewmode"))
		{
			const std::string viewmode = vm_["viewmode"].as<std::string>();
			if (boost::iequals(viewmode, "magnitude"))
				opts_.cd.img_type = ImgType::Modulus;
			else if (boost::iequals(viewmode, "squaredmagnitude"))
				opts_.cd.img_type = ImgType::SquaredModulus;
			else if (boost::iequals(viewmode, "argument"))
				opts_.cd.img_type = ImgType::Argument;
			else
				throw std::runtime_error("unknown view mode");
		}

		opts_.cd.log_scale_slice_xy_enabled = vm_.count("log") > 0;

		opts_.cd.fft_shift_enabled = vm_.count("nofftshift");

		if (vm_.count("contrastmin"))
		{
			const double log_min = vm_["contrastmin"].as<double>();

			if (log_min < -100.0f || log_min > 100.0f)
				throw std::runtime_error("wrong min parameter (-100.0 < min < 100.0)");

			if (opts_.cd.log_scale_slice_xy_enabled)
				opts_.cd.contrast_min_slice_xy = static_cast<float>(log_min);
			else
				opts_.cd.contrast_min_slice_xy = static_cast<float>(pow(10.0, log_min));
			opts_.cd.contrast_enabled = true;
		}

		if (vm_.count("contrastmax"))
		{
			const float log_max = vm_["contrastmax"].as<float>();

			if (log_max < -100.0f || log_max > 100.0f)
				throw std::runtime_error("wrong max parameter (-100.0 < max < 100.0)");

			if (opts_.cd.log_scale_slice_xy_enabled)
				opts_.cd.contrast_max_slice_xy = log_max;
			else
				opts_.cd.contrast_max_slice_xy = static_cast<float>(pow(10.0, log_max));
			opts_.cd.contrast_enabled = true;
		}
	}

	void OptionsParser::check_compute_params()
	{
		if (!vm_.count("nSize"))
			throw std::runtime_error("--nSize is required");

		if (!vm_.count("pindex"))
			throw std::runtime_error("--pindex is required");

		if (!vm_.count("lambda"))
			throw std::runtime_error("--lambda is required");

		if (!vm_.count("zdistance"))
			throw std::runtime_error("--zdistance is required");
	}
}