#include "options_parser.hh"

#include <boost\algorithm\string.hpp>
#include <boost\lexical_cast.hpp>

#include <cassert>

namespace holovibes
{
  OptionsParser::OptionsParser(OptionsDescriptor& opts)
    : opts_(opts)
    , pos_desc_()
    , general_opts_desc_("General")
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
      ;
  }

  void OptionsParser::init_features_options(bool is_no_gui)
  {
    features_opts_desc_.add_options()
      ("queuesize,q",
      po::value<int>()
      ->default_value(default_queue_size),
      "Size of queue arg in number of images")
      ;

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

        ("cameramodel,c",
        po::value<std::string>()
        ->required(),
        "Set the camera to use: pike/xiq/ids/pixelfly/ixon/edge.")
        ;
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

      ("nsamples,n",
      po::value<int>(),
      "Number of samples N.")

      ("pindex,p",
      po::value<int>(),
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

      ("vibrometry,v",
      po::value<int>(),
      "Select the q-th component of the DFT and enable vibrometry, vq must be defined in {0, ..., N - 1}.")
      ;
  }

  void OptionsParser::init_merge_options()
  {
    merge_opts_desc_.add(general_opts_desc_);
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

      init_features_options(!opts_.is_gui_enabled);
      init_merge_options();

      /* May exit here. */
      proceed_help();

      parse_features_compute_options(argc, argv);

      proceed_features();
      proceed_compute();

      if (!opts_.is_gui_enabled && opts_.is_compute_enabled)
        check_compute_params();

      opts_.compute_desc.sanity_check();

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
    std::cout << "\nUsage: ./holovibes.exe [OPTIONS]\n"
      << "This help message depends on --nogui parameter (more options available).\n" << std::endl;
    std::cout << merge_opts_desc_;
  }

  void OptionsParser::print_version()
  {
    std::cout << "\nHolovibes " << version << std::endl;
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
        opts_.camera = Holovibes::XIQ;
      else if (boost::iequals(camera, "ids"))
        opts_.camera = Holovibes::IDS;
      else if (boost::iequals(camera, "pike"))
        opts_.camera = Holovibes::PIKE;
      else if (boost::iequals(camera, "pixelfly"))
        opts_.camera = Holovibes::PIXELFLY;
      else if (boost::iequals(camera, "ixon"))
        opts_.camera = Holovibes::IXON;
      else if (boost::iequals(camera, "edge"))
        opts_.camera = Holovibes::EDGE;
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
        /* This case should not append. */
        assert(!"Display vector<int> is empty");
      }

      opts_.is_gl_window_enabled = true;
    }

    if (vm_.count("queuesize"))
    {
      const int queue_size = vm_["queuesize"].as<int>();

      if (queue_size > 0)
        opts_.queue_size = queue_size;
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
  }

  void OptionsParser::proceed_compute()
  {
    if (vm_.count("1fft"))
    {
      opts_.is_compute_enabled = true;
      opts_.compute_desc.algorithm = ComputeDescriptor::FFT1;
    }

    if (vm_.count("2fft"))
    {
      if (opts_.is_compute_enabled)
        throw std::runtime_error("1fft method already selected");

      opts_.is_compute_enabled = true;
      opts_.compute_desc.algorithm = ComputeDescriptor::FFT2;
    }

    if (vm_.count("nsamples"))
    {
		const unsigned short nsamples = vm_["nsamples"].as<unsigned short>();
      if (nsamples <= 0)
        throw std::runtime_error("--nsamples parameter must be strictly positive");

      opts_.compute_desc.nsamples = nsamples;

      if (opts_.compute_desc.nsamples >= opts_.queue_size)
        throw std::runtime_error("--nsamples can not be greater than the queue size");
    }

    if (vm_.count("pindex"))
    {
		const unsigned short pindex = vm_["pindex"].as<unsigned short>();
      if (pindex < 0 || pindex >= opts_.compute_desc.nsamples)
        throw std::runtime_error("--pindex parameter must be defined in {0, ..., nsamples - 1}.");
      opts_.compute_desc.pindex = pindex;
    }

    if (vm_.count("lambda"))
    {
      const float lambda = vm_["lambda"].as<float>();
      if (lambda <= 0.0000f)
        throw std::runtime_error("--lambda parameter must be strictly positive");
      opts_.compute_desc.lambda = lambda;
    }

    if (vm_.count("zdistance"))
    {
      const float zdistance = vm_["zdistance"].as<float>();
      opts_.compute_desc.zdistance = zdistance;
    }

    if (vm_.count("viewmode"))
    {
      const std::string viewmode = vm_["viewmode"].as<std::string>();
      if (boost::iequals(viewmode, "magnitude"))
        opts_.compute_desc.view_mode = ComputeDescriptor::MODULUS;
      else if (boost::iequals(viewmode, "squaredmagnitude"))
        opts_.compute_desc.view_mode = ComputeDescriptor::SQUARED_MODULUS;
      else if (boost::iequals(viewmode, "argument"))
        opts_.compute_desc.view_mode = ComputeDescriptor::ARGUMENT;
      else
        throw std::runtime_error("unknown view mode");
    }

    opts_.compute_desc.log_scale_enabled = vm_.count("log");

    opts_.compute_desc.shift_corners_enabled = !vm_.count("nofftshift");

    if (vm_.count("contrastmin"))
    {
		const double log_min = vm_["contrastmin"].as<double>();

      if (log_min < -100.0f || log_min > 100.0f)
        throw std::runtime_error("wrong min parameter (-100.0 < min < 100.0)");

      if (opts_.compute_desc.log_scale_enabled)
        opts_.compute_desc.contrast_min = log_min;
      else
        opts_.compute_desc.contrast_min = pow(10.0, log_min);
      opts_.compute_desc.contrast_enabled = true;
    }

    if (vm_.count("contrastmax"))
    {
      const float log_max = vm_["contrastmax"].as<float>();

      if (log_max < -100.0f || log_max > 100.0f)
        throw std::runtime_error("wrong max parameter (-100.0 < max < 100.0)");

      if (opts_.compute_desc.log_scale_enabled)
        opts_.compute_desc.contrast_max = log_max;
      else
        opts_.compute_desc.contrast_max = pow(10.0, log_max);
      opts_.compute_desc.contrast_enabled = true;
    }

    if (vm_.count("vibrometry"))
    {
      const int vibrometry_q = vm_["vibrometry"].as<int>();
      if (vibrometry_q < 0 || static_cast<unsigned int>(vibrometry_q) >= opts_.compute_desc.nsamples)
        throw std::runtime_error("--vibrometry parameter must be defined in {0, ..., nsamples - 1}.");
      opts_.compute_desc.vibrometry_q = vibrometry_q;
      opts_.compute_desc.vibrometry_enabled = true;
    }
  }

  void OptionsParser::check_compute_params()
  {
    if (!vm_.count("nsamples"))
      throw std::runtime_error("--nsamples is required");

    if (!vm_.count("pindex"))
      throw std::runtime_error("--pindex is required");

    if (!vm_.count("lambda"))
      throw std::runtime_error("--lambda is required");

    if (!vm_.count("zdistance"))
      throw std::runtime_error("--zdistance is required");
  }
}
