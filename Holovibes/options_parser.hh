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
    const std::string version = "v.0.5.4";
    const unsigned int default_queue_size = 20;
    const unsigned int display_size_min = 100;

  public:
    OptionsParser(OptionsDescriptor& opts);

    /*! \brief Parse the command line given by the user and
     * fill the options descriptor. Will automatically call
     * help/version print and exit.
     * ComputeDescriptor::sanity_check is called. */
    void parse(int argc, char* const argv[]);

  private:
    /*! Parser initialization: loads options. */
    void init_general_options();
    void init_features_options(bool is_no_gui);
    void init_compute_options();
    /*! Build merge descriptor that contains
     * general/features/compute options. */
    void init_merge_options();

    /*! Parse the command line with general options
     * descriptor. */
    void parse_general_options(int argc, char* const argv[]);
    /*! Is gui enabled regarding the nogui option. */
    bool get_is_gui_enabled();

    void parse_features_compute_options(int argc, char* const argv[]);

    /*! Handle help/version options. */
    void proceed_help();
    /*! Check features values and fill OptionsDescriptor.*/
    void proceed_features();
    /*! Fill OptionsDescriptor with compute values. */
    void proceed_compute();

    /*! Checks DFT parameters. Each parameter is mandatory. */
    void check_compute_params();

    /*! Print the version & help message. */
    void print_help();
    /*! Print the version message. */
    void print_version();
  private:
    /* Contains all Holovibes' options. */
    OptionsDescriptor& opts_;

    /* Empty positional options. */
    const po::positional_options_description pos_desc_;
    /*! Describes general program options. */
    po::options_description general_opts_desc_;
    /*! Describes features (what to do) options of Holovibes. */
    po::options_description features_opts_desc_;
    /*! Describes compute (how to do) options (cuda, fft, ...) */
    po::options_description compute_opts_desc_;
    /*! Contains general/features/compute options descriptors. */
    po::options_description merge_opts_desc_;
    /*! Stores all option values. */
    po::variables_map vm_;

    OptionsParser& operator=(const OptionsParser&) = delete;
  };
}
#endif /* !OPTIONS_PARSER_HH */
