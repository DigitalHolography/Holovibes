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
#include <boost/fusion/include/std_pair.hpp>
#include <boost/fusion/support/pair.hpp>
#include <boost/fusion/include/pair.hpp>
#include <boost/tuple/tuple.hpp> 
#include <boost/tuple/tuple_io.hpp>
#include <iostream>
#include <fstream>

namespace holovibes
{
  typedef struct options
  {
    int nbimages;
    std::string record_path;
    int buffsize;
    int set_size;
    int width;
    int height;
    int width_win;
    int height_win;
    bool display_images;
    bool record;
    std::string cam;
  } s_options;

  class OptionParser
  {
  public:
    OptionParser(const int argc, const char** argv)
      : argc_(argc)
      , argv_(argv)
      , help_(false)
      , version_(false)
    {}

    void init_parser(); //initialize the parser must be called right after the creation
    void proceed(); //parse the command line given by the user and fill the options_ strucure
    s_options& get_opt()
    {
      return options_;
    };

  private:

    void proceed_help();
    void proceed_display();
    void proceed_buffsize();
    void proceed_imageset();
    void proceed_frameinfo();
    void proceed_version();
    void proceed_win_size();
    void proceed_cam();
    void check_integrity();

    OptionParser& operator=(const OptionParser&) = delete;
    const int argc_;
    const char** argv_;
    bool help_;
    bool version_;

    boost::program_options::options_description desc_;
    boost::program_options::variables_map vm_;
    s_options options_;
  };
}
#endif