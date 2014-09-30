#ifndef OPTION_PARSER_HH
# define OPTION_PARSER_HH


#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/eof_iterator.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/environment_iterator.hpp>
#include <boost/program_options/config.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/version.hpp>
#include <iostream>
#include <fstream>

typedef struct options
{
  int nbimages;
  int buffsize;
  int set_size;
  int width;
  int height;
  int bitdepth;
  int binning;
  bool display_images;
} options_s;


class Option_parser
{
public:
  Option_parser(int argc, char** argv);
  void init_parser();
  void proceed_help();
  void proceed_nbimages();
  void proceed_display();
  void proceed_buffsize();
  void proceed_imageset();
  void proceed_frameinfo();
  void proceed_binning();
  void proceed_version();
  void proceed();

private:
  boost::program_options::options_description desc_;
  bool help_;
  bool version_;
  int argc_;
  char** argv_;
  boost::program_options::variables_map vm_;
  options_s options_;
};





#endif