#include "stdafx.h"
#include "option_parser.hh"

using namespace boost;
using namespace boost::program_options;

Option_parser::Option_parser(int argc, char** argv)
{
  argc_ = argc;
  argv_ = argv;
  help_ = false;
}

void Option_parser::init_parser()
{
  desc_.add_options()
    ("help", "Produce help message")
    ("nbimages", program_options::value<int>(), "Set the number of images to acquire until the program is closed")
    ("display", "Display the images captured from the camera")
    ("record", program_options::value<std::string>(), "Record the images from the camera, wait for arg the path where to save the file")
    ("buffersize", program_options::value<int>(), "Give the size of the internal buffer allocated in the Ram to store iomages, default 50")
    ("imageset", program_options::value<int>(), "Set the size of the set of images to write on the record file. this number should be less than the buffer size")
    ("width", program_options::value<int>(), "Set the width value of the frame to capture to arg value")
    ("height", program_options::value<int>(), "Set the height value of the frame to capture to arg value")
    ("bitdepth", program_options::value<int>(), "Set the bitedepth of the frame to capture to arg value")
    ("binning", program_options::value<int>(), "Set the binning mode")
    ;

 
    program_options::store(program_options::parse_command_line(argc_, argv_, desc_), vm_);
    program_options::notify(vm_);

}

void Option_parser::proceed_help()
{
  if (vm_.count("help"))
  {
    std::cout << desc_ << std::endl;
    help_ = true;
  }
}

void Option_parser::proceed_display()
{
  if (vm_.count("display"))
  {
    std::cout << "Images will be displayed" << std::endl;
    options_.display_images = true;
  }
  else if (vm_.count("record"))
  {
    std::cout << "The images will be recorded to " <<
      vm_["record"].as<std::string>() << std::endl;
    options_.display_images = false;
  }
}

void Option_parser::proceed_frameinfo()
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

void Option_parser::proceed_nbimages()
{
  if (vm_.count("nbimages"))
  {
    std::cout << "images to save was set to " <<
      vm_["nbimages"].as<int>() << std::endl;
    options_.nbimages = vm_["nbimages"].as<int>();
  }
}

void Option_parser::proceed_buffsize()
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

void Option_parser::proceed_imageset()
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

void Option_parser::proceed_binning()
{
  if (vm_.count("binning"))
  {
    std::cout << "The binning mode is set to " <<
      vm_["binning"].as<int>() << std::endl;
    options_.binning = vm_["binning"].as<int>();
  }
}

void Option_parser::proceed()
{
  proceed_help();
  if (!help_)
  {
    proceed_nbimages();
    proceed_display();
    proceed_buffsize();
    if (!options_.display_images)
      proceed_imageset();
    proceed_frameinfo();
  }
  help_ = false;
}