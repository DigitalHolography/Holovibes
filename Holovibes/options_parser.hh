#ifndef OPTIONS_PARSER_HH
# define OPTIONS_PARSER_HH

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
    const std::string version = "v0.2";
    const unsigned int default_queue_size = 20;
    const unsigned int display_size_min = 100;

  public:
    OptionsParser(OptionsDescriptor& opts)
      : opts_(opts)
      , pos_desc_()
      , help_desc_("Help")
      , cuda_desc_("CUDA options")
      , desc_("General options")
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
    /*! Checks DFT parameters. Each parameter is mandatory. */
    void proceed_dft_params();

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
    /*! Describes compute options (cuda, fft, ...) */
    po::options_description cuda_desc_;
    /*! Describes all program options. */
    po::options_description desc_;
    /*! Stores all option values. */
    po::variables_map vm_;

    OptionsParser& operator=(const OptionsParser&) = delete;
  };
}
#endif /* !OPTIONS_PARSER_HH */
