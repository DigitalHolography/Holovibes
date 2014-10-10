#ifndef OPTIONS_PARSER_HH
# define OPTIONS_PARSER_HH

// Clean all these includes.
# include <boost/tokenizer.hpp>
# include <boost/program_options/options_description.hpp>
# include <boost/program_options/cmdline.hpp>
# include <boost/program_options/eof_iterator.hpp>
# include <boost/program_options/errors.hpp>
# include <boost/program_options/option.hpp>
# include <boost/program_options/parsers.hpp>
# include <boost/program_options/variables_map.hpp>
# include <boost/program_options/positional_options.hpp>
# include <boost/program_options/environment_iterator.hpp>
# include <boost/program_options/config.hpp>
# include <boost/program_options/value_semantic.hpp>
# include <boost/program_options/version.hpp>

# include "options_descriptor.hh"

/* Boost namespaces alias. */
namespace po = boost::program_options;

namespace holovibes
{
  class OptionsParser
  {
    const std::string version = "v0.1";
    const int default_queue_size = 50;
    const int default_img_set_size = 10;
    const int display_size_min = 100;

  public:
    OptionsParser(OptionsDescriptor& opts)
      : opts_(opts)
      , pos_desc_()
      , help_desc_("Help")
      , desc_("Holovibes options")
      , vm_()
    {
      init_parser();
    }

    /*! \brief Parse the command line given by the user and
     * fill the options descriptor. */
    void parse(int argc, const char* argv[]);

  private:
    /*! Parser initialization: loads options. */
    void init_parser();
    /*! Check values and fill OptionsDescriptor.*/
    void proceed_holovibes();
    /*! Handle help/version options. */
    void proceed_help();

    /*! Print the version & help message. */
    void print_help();
    /*! Print the version message. */
    void print_version();
  private:
    OptionsDescriptor& opts_;

    /* Empty positional options. */
    const po::positional_options_description pos_desc_;
    /*! Describes help program options. */
    po::options_description help_desc_;
    /*! Describes all program options. */
    po::options_description desc_;
    /*! Stores all option values. */
    po::variables_map vm_;

    OptionsParser& operator=(const OptionsParser&) = delete;
  };
}
#endif /* !OPTIONS_PARSER_HH */
