/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Options parser for the command-line. */
#pragma once

#include <optional>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

namespace po = boost::program_options;

namespace holovibes
{
struct OptionsDescriptor
{
    bool print_help;
    bool print_version;
    bool disable_gui;
    std::optional<unsigned int> fps;
    std::optional<unsigned int> n_rec;
    std::optional<std::string> input_path;
    std::optional<std::string> output_path;
    bool record_raw;
};

/*! \brief Options parser for the command-line. */
class OptionsParser
{
  public:
    OptionsParser();

    virtual ~OptionsParser() = default;

    OptionsParser& operator=(const OptionsParser&) = delete;

    /*! \brief Parse the command line given by the user and
     * fill the options descriptor. Will automatically call
     * help/version print and exit. */
    OptionsDescriptor parse(int argc, char* const argv[]);

    po::options_description get_opts_desc() const;

  private:
    /*! \brief Describes all program options. */
    po::options_description opts_desc_;

    /*! \brief Stores all option values. */
    po::variables_map vm_;

    OptionsDescriptor options_;
};
} // namespace holovibes
