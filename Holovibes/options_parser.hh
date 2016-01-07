#pragma once

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

namespace po = boost::program_options;

/* Forward declaration. */
namespace holovibes
{
  class OptionsDescriptor;
}

namespace holovibes
{
  /*! \brief Parse Argv gived to program */
  class OptionsParser
  {
    /*! \{ \name const */
    const unsigned int default_queue_size = 20;
    const unsigned int display_size_min = 100;
    const unsigned int default_fps = 30;
    const unsigned int default_depth = 8;
    const bool default_is_big_endian = true;
    /*! \} */

  public:
    OptionsParser(OptionsDescriptor& opts);

    OptionsParser& operator=(const OptionsParser&) = delete;

    /*! \brief Parse the command line given by the user and
     * fill the options descriptor. Will automatically call
     * help/version print and exit. */
    void parse(int argc, char* const argv[]);

  private:
    /*! \{ \name Init options*/
    /*! \brief Parser initialization: loads program options. */
    void init_general_options();

    /*! \brief Parser initialization: loads features options. */
    void init_features_options(bool is_no_gui, bool is_import_mode_enable);

    /*! \brief Parser initialization: loads compute options. */
    void init_compute_options();

    /*! \brief Build merge descriptor that contains
     * general/features/compute options. */
    void init_merge_options();
    /*! \} */ // End of Init group

    /*! \brief Parse the command line with general options
     * descriptor. */
    void parse_general_options(int argc, char* const argv[]);

    /*! \brief Is gui enabled regarding the nogui option. */
    bool get_is_gui_enabled();

    /*! \brief Is import mode enabled regarding the import option. */
    bool get_is_import_mode();

    void parse_features_compute_options(int argc, char* const argv[]);

    /*! \brief Handle help/version options. */
    void proceed_help();

    /*! \brief Check features values and fill OptionsDescriptor.*/
    void proceed_features();

    /*! \brief Fill OptionsDescriptor with compute values. */
    void proceed_compute();

    /*! \brief Checks DFT parameters. Each parameter is mandatory. */
    void check_compute_params();

    /*! \brief Print the version & help message. */
    void print_help();

    /*! \brief Print the version message. */
    void print_version();

  private:
    /*! \brief Contains all Holovibes' options. */
    OptionsDescriptor& opts_;

    /*! \brief Empty positional options. */
    const po::positional_options_description pos_desc_;
    /*! \brief Describes general program options. */
    po::options_description general_opts_desc_;
    /*! \brief Describes import mode (what look file). */
    po::options_description import_opts_desc_;
    /*! \brief Describes features (what to do) options of Holovibes. */
    po::options_description features_opts_desc_;
    /*! \brief Describes compute (how to do) options (cuda, fft, ...) */
    po::options_description compute_opts_desc_;
    /*! \brief Contains general/features/compute options descriptors. */
    po::options_description merge_opts_desc_;

    /*! \brief Stores all option values. */
    po::variables_map vm_;
  };
}