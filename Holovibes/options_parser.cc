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

      ("fft1,f",
      po::value<std::vector<std::string>>()->multitoken(),
      "p, set size, lambda, dist"
      )
      ;
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
    if (vm_.count("fft1"))
    {
      const std::vector<std::string>& args =
        vm_["fft1"].as<std::vector<std::string>>();
      if (args.size() == 4)
      {
        try
        {
          int p = boost::lexical_cast<int>(args[0]);
          int nbimage = boost::lexical_cast<int>(args[1]);
          float lambda = boost::lexical_cast<float>(args[2]);
          float dist = boost::lexical_cast<float>(args[3]);
          opts_.nbimages = nbimage;
          opts_.distance = dist;
          opts_.lambda = lambda;
          opts_.p = p;
          std::cout << "p: " << p << " nbi: " << nbimage << " lambda: " << lambda << " dist: " << dist << std::endl;
        }
        catch (boost::bad_lexical_cast&)
        {
          throw std::exception("wrong fft parameters (must be numbers)");
        }
      }
      else
        throw std::exception("-f/--fft expects 4 arguments");
    }

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
  }
}
