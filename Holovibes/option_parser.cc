#include "stdafx.h"
#include "option_parser.hh"

using namespace boost;
using namespace boost::program_options;
namespace holovibes
{
  void OptionParser::init_parser()
  {
    desc_.add_options()
      ("help", "Produce help message")
      ("nbimages", program_options::value<int>(), "Set the number of images to acquire until the program is closed")
      ("display", "Display the images captured from the camera")
      ("record", program_options::value<std::string>(), "Record the images from the camera, wait for arg the path where to save the file")
      ("buffersize", program_options::value<int>(), "Give the size of the internal buffer allocated in the RAM to store images, default 50")
      ("imageset", program_options::value<int>(), "Set the size of the set of images to write on the record file. this number should be less than the buffer size")
      ("width", program_options::value<int>(), "Set the width value of the frame to capture to arg value")
      ("height", program_options::value<int>(), "Set the height value of the frame to capture to arg value")
      ("bitdepth", program_options::value<int>(), "Set the bit depth of the frame to capture to arg value")
      ("binning", program_options::value<int>(), "Set the binning mode")
      ("version", "Display the version of the release used")
      ("widthwin", program_options::value<int>(), "Set the width value of the frame to capture to arg value")
      ("heightwin", program_options::value<int>(), "Set the height value of the frame to capture to arg value")
      ("cam", program_options::value<std::string>(), "Set the camera to use: pike/xiq/ids")
      ;

    try
    {
      program_options::store(program_options::parse_command_line(argc_, argv_, desc_), vm_);
    }
    catch (boost::program_options::invalid_option_value &e_)
    {
      std::cout << desc_ << std::endl;
      help_ = true;
      std::cout << "WARNING: One of your option(s) argument does not comply to any of the one below, please refer to the help page" << std::endl << std::endl;
    }
    catch (...)
    {
      std::cout << desc_ << std::endl;
      help_ = true;
      std::cout << "WARNING: One of your option(s) does not refer to any of the one below, please refer to the help page" << std::endl << std::endl;
    }
    program_options::notify(vm_);
  }

  void OptionParser::proceed_help()
  {
    if (vm_.count("help"))
    {
      std::cout << desc_ << std::endl;
      help_ = true;
    }
  }

  void OptionParser::proceed_cam()
  {
    if (vm_.count("cam"))
    {
      std::cout << "The choosen camera is " <<
        vm_["cam"].as<std::string>() << std::endl;
    }
  }

  s_options OptionParser::get_opt()
  {
    return options_;
  }

  void OptionParser::proceed_display()
  {
    if (vm_.count("display"))
    {
      std::cout << "Images will be displayed" << std::endl;
      options_.display_images = true;
      options_.record = false;
    }
    else if (vm_.count("record"))
    {
      std::cout << "The images will be recorded to " <<
        vm_["record"].as<std::string>() << std::endl;
      options_.display_images = false;
      options_.record = true;
      options_.record_path = vm_["record"].as<std::string>();
    }
  }

  void OptionParser::proceed_frameinfo()
  {
    if (vm_.count("width"))
    {
      std::cout << "Images width is " <<
        vm_["width"].as<int>() << std::endl;
      options_.width = vm_["width"].as<int>();
    }
    if (vm_.count("height"))
    {
      std::cout << "Images height is " <<
        vm_["height"].as<int>() << std::endl;
      options_.height = vm_["height"].as<int>();
    }
    if (vm_.count("bitdepth"))
    {
      std::cout << "Pixel bitdepth is " <<
        vm_["bitdepth"].as<int>() << std::endl;
      options_.bitdepth = vm_["bitdepth"].as<int>();
    }
  }

  void OptionParser::proceed_win_size()
  {
    if (vm_.count("widthwin"))
    {
      std::cout << "Window width is " <<
        vm_["widthwin"].as<int>() << std::endl;
      options_.width_win = vm_["widthwin"].as<int>();
    }
    if (vm_.count("heightwin"))
    {
      std::cout << "Window height is " <<
        vm_["heightwin"].as<int>() << std::endl;
      options_.height_win = vm_["heightwin"].as<int>();
    }
  }

  void OptionParser::proceed_nbimages()
  {
    if (vm_.count("nbimages"))
    {
      std::cout << "images to save was set to " <<
        vm_["nbimages"].as<int>() << std::endl;
      options_.nbimages = vm_["nbimages"].as<int>();
    }
  }

  void OptionParser::proceed_buffsize()
  {
    if (vm_.count("buffersize"))
    {
      std::cout << "The buffer size is set to " <<
        vm_["buffersize"].as<int>() << std::endl;
      options_.buffsize = vm_["buffersize"].as<int>();
    }
    else
    {
      options_.buffsize = 100;
      std::cout << "The buffer size has been set to " <<
        options_.buffsize << " by default" << std::endl;
    }
  }

  void OptionParser::proceed_imageset()
  {
    if (vm_.count("imageset"))
    {
      std::cout << "The image set size is set to " <<
        vm_["imageset"].as<int>() << std::endl;
      options_.set_size = vm_["imageset"].as<int>();
    }
    else
    {
      options_.set_size = 10;
      std::cout << "The image set size has been set to " <<
        options_.set_size << " by default" << std::endl;
    }
  }

  void OptionParser::proceed_binning()
  {
    if (vm_.count("binning"))
    {
      std::cout << "The binning mode is set to " <<
        vm_["binning"].as<int>() << std::endl;
      options_.binning = vm_["binning"].as<int>();
    }
  }

  void OptionParser::proceed_version()
  {
    if (vm_.count("version"))
    {
      std::cout << "Holovibes  v1.O" << std::endl;
      version_ = true;
    }
  }

  void OptionParser::proceed()
  {
    proceed_version();
    proceed_help();
    if (!help_ && !version_)
    {
      proceed_win_size();
      proceed_nbimages();
      proceed_display();
      proceed_binning();
      proceed_cam();
      if ((!options_.display_images && options_.record)
        || (options_.display_images && !options_.record))
        proceed_imageset();
      if (options_.record)
        proceed_buffsize();
      proceed_frameinfo();
    }
    help_ = false;
  }
}