#include "stdafx.h"
#include "options_parser.hh"

#include <boost\algorithm\string.hpp>
#include <boost\lexical_cast.hpp>

#include <cassert>

namespace holovibes
{
  void OptionsParser::init_parser()
  {
    help_desc_.add_options()
      ("version", "Print the version number of Holovibes and exit.")
      ("help,h", "Print a summary of the command-line options to Holovibes and exit.")
      ;

    desc_.add_options()
      ("display,d",
      po::value<std::vector<int>>()
      ->multitoken(),
      "Display images on screen. "
      "The first argument gives the square size of the display. "
      "The second optional argument specify the height.")

      ("write,w",
      po::value<std::vector<std::string>>()
      ->multitoken(),
      "Record a sequence of images in the given path. "
      "The first argument gives the number of images to record. "
      "The second argument gives the filepath where frames will be recorded.")

      ("queuesize,q",
      po::value<int>()
      ->default_value(default_queue_size),
      "Size of queue arg in number of images")

      ("cameramodel,c",
      po::value<std::string>()
      ->required(),
      "Set the camera to use: pike/xiq/ids/pixelfly.")
      ;

    cuda_desc_.add_options()
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
      ;

    desc_.add(cuda_desc_);
    desc_.add(help_desc_);
  }

  void OptionsParser::parse(int argc, const char* argv[])
  {
    bool succeed = false;

    try
    {
      /* First parsing to check help/version options. */
      po::store(po::command_line_parser(argc, argv)
        .options(help_desc_)
        .allow_unregistered()
        .run(), vm_);
      po::notify(vm_);
      proceed_help();

      /* Parsing holovibes options. */
      po::store(
        po::command_line_parser(argc, argv)
        .options(desc_)
        .positional(pos_desc_)
        .run(), vm_);
      po::notify(vm_);
      proceed_holovibes();
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
    catch (std::exception &e)
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
    std::cout << "\nUsage: ./holovibes.exe [OPTIONS]" << std::endl;
    std::cout << desc_;
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

  void OptionsParser::proceed_holovibes()
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
      else
        throw std::exception("unknown camera model");
    }

    if (vm_.count("display"))
    {
      const std::vector<int>& display_size =
        vm_["display"].as<std::vector<int>>();

      if (!display_size.empty())
      {
        /* Display size check. */
        if (display_size[0] < display_size_min ||
          display_size.size() >= 2 && (display_size[1] < display_size_min))
        {
          throw std::exception("display width/height is too small (<100)");
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
        throw std::exception("queue size is too small");
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
            throw std::exception("record filepath is empty");

          opts_.recorder_n_img = n_img;
          opts_.recorder_filepath = filepath;
        }
        catch (boost::bad_lexical_cast&)
        {
          throw std::exception("wrong record first parameter (must be a number)");
        }
      }
      else
        throw std::exception("-w/--write expects 2 arguments");

      opts_.is_recorder_enabled = true;
    }

    if (vm_.count("1fft"))
    {
      proceed_dft_params();
      opts_.is_1fft_enabled = true;
      opts_.compute_desc.algorithm = ComputeDescriptor::FFT1;
    }

    if (vm_.count("2fft"))
    {
      if (opts_.is_1fft_enabled)
        throw std::exception("1fft method already selected");

      proceed_dft_params();
      opts_.is_2fft_enabled = true;
      opts_.compute_desc.algorithm = ComputeDescriptor::FFT2;
    }
  }

  void OptionsParser::proceed_dft_params()
  {
    if (vm_.count("nsamples"))
    {
      const int nsamples = vm_["nsamples"].as<int>();

      if (nsamples <= 0)
        throw std::exception("--nsamples parameter must be strictly positive");

      if (nsamples >= opts_.queue_size)
        throw std::exception("--nsamples can not be greater than the queue size");

      opts_.compute_desc.nsamples = nsamples;
    }
    else
      throw std::exception("--nsamples is required");

    if (vm_.count("pindex"))
    {
      const int pindex = vm_["pindex"].as<int>();

      if (pindex < 0 || pindex >= opts_.compute_desc.nsamples)
        throw std::exception("--pindex parameter must be defined in {0, ..., nsamples - 1}.");

      opts_.compute_desc.pindex = pindex;
    }
    else
      throw std::exception("--pindex is required");

    if (vm_.count("lambda"))
    {
      const float lambda = vm_["lambda"].as<float>();

      if (lambda <= 0.0000f)
        throw std::exception("--lambda parameter must be strictly positive");

      opts_.compute_desc.lambda = lambda;
    }
    else
      throw std::exception("--lambda is required");

    if (vm_.count("zdistance"))
    {
      const float zdistance = vm_["zdistance"].as<float>();

      opts_.compute_desc.zdistance = zdistance;
    }
    else
      throw std::exception("--zdistance is required");
  }
}
