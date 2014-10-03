#include "stdafx.h"
#include "option_parser.hh"

using namespace boost;
using namespace boost::program_options;
namespace holovibes
{
  void OptionParser::init_parser()
  {
    // -42 is a magic number to check the integrity of given params
    options_.set_size = 0;
    options_.buffsize = 0;
    options_.height_win = 800;
    options_.width_win = 800;
    options_.nbimages = 0;
    options_.width = 0;
    options_.height = 0;

    desc_.add_options()
      ("help,h", "Help message")
      ("display,d", program_options::value<std::vector<int>>(), "Display images on screen")
      ("record,r", program_options::value<std::vector<std::string>>(), "Record  a sequence of images in the given path")
      ("queuesize,q", program_options::value<int>(), "Size of queue arg in number of images")
      ("imagesetsize,i", program_options::value<int>(), "Set the size of the set of images to write on the record file. this number should be less than the buffer size")
      ("width,w", program_options::value<int>(), "Set the width value of the frame to capture to arg value")
      ("height,h", program_options::value<int>(), "Set the height value of the frame to capture to arg value")
      ("version", "Display the version of the release used")
      ("cameramodel,c", program_options::value<std::string>(), "Set the camera to use: pike/xiq/ids")
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
    if (vm_.count("cameramodel"))
    {
      std::cout << "The choosen camera is " <<
        vm_["cameramodel"].as<std::string>() << std::endl;
      options_.cam = vm_["cameramodel"].as<std::string>();
    }
  }
  void OptionParser::proceed_display()
  {
    if (vm_.count("display"))
    {
      std::cout << "Images will be displayed" << std::endl;
      options_.display_images = true;
      options_.record = false;
      std::vector<int> tmp = vm_["display"].as<std::vector<int>>();
      if (tmp.size() == 1)
      {
        options_.height_win = tmp[0];
        options_.width_win = tmp[0];
      }
      else if (tmp.size() > 1)
      {
        options_.height_win = tmp[0];
        options_.width_win = tmp[1];
      }
    }
    else if (vm_.count("record"))
    {
      options_.record = true;
      std::cout << "into record opt" << std::endl;//FIXME
      std::vector<std::string> tmp = vm_["record"].as<std::vector<std::string>>();
      std::cout << "vec size " << tmp.size() << std::endl; //FIXME
      std::cout << "first mermber" << tmp[0] << std::endl; //FIXME
      if (tmp.size() > 1)
      {
        std::cout << "into record opt vector looks good" << std::endl;//FIXME
        options_.nbimages = atoi(tmp[0].c_str());
        options_.record_path = tmp[1];
        std::cout << "The images will be recorded to " <<
          options_.record_path << std::endl;
        std::cout << options_.nbimages 
          << "The images will be recorded"<< std::endl;

      }
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
  }

  void OptionParser::proceed_win_size()
  {

    {
      std::cout << "Window height is " <<
        800 << " by default" << std::endl;
      options_.height_win = 800;
    }

  }

  void OptionParser::proceed_buffsize()
  {
    if (vm_.count("queuesize"))
    {
      std::cout << "The queue size is set to " <<
        vm_["queuesize"].as<int>() << std::endl;
      options_.buffsize = vm_["queuesize"].as<int>();
    }
    else
    {
      options_.buffsize = 100;
      std::cout << "The queue size has been set to " <<
        options_.buffsize << " by default" << std::endl;
    }
  }

  void OptionParser::proceed_imageset()
  {
    if (vm_.count("imagesetsize"))
    {
      std::cout << "The image set size is set to " <<
        vm_["imagesetsize"].as<int>() << std::endl;
      options_.set_size = vm_["imagesetsize"].as<int>();
    }
    else
    {
      options_.set_size = 10;
      std::cout << "The image set size has been set to " <<
        options_.set_size << " by default" << std::endl;
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

  void OptionParser::check_integrity()
  {
    if (options_.cam == "")
      std::cerr << ("Please specify a camera to use") << std::endl;
    if (options_.buffsize <= 0)
      std::cerr << ("Please specify a correct queue size") << std::endl;;
    if (options_.set_size <= 0)
      std::cerr << ("Please specify a correct imageset size") << std::endl;;
    if (options_.record && (options_.nbimages <= 0))
      std::cerr << ("Please specify a number of images to record") << std::endl;
  }

  void OptionParser::proceed()
  {
    proceed_version();
    proceed_help();
    if (!help_ && !version_)
    {
      proceed_display();
      proceed_cam();
      if ((!options_.display_images && options_.record)
        || (options_.display_images && !options_.record))
        proceed_imageset();
      if (options_.record)
        proceed_buffsize();
      if (options_.display_images)
        proceed_win_size();
      proceed_frameinfo();
    }
    help_ = false;
    check_integrity();
  }
}