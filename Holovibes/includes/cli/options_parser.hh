/*! \file
 *
 * \brief Options parser for the command-line.
 */
#pragma once

#include <optional>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

namespace po = boost::program_options;

namespace holovibes
{
/*! \struct OptionsDescriptor
 *
 * \brief Struct used for option description
 */
struct OptionsDescriptor
{
    bool print_help;
    bool print_version;
    bool disable_gui;
    std::optional<unsigned int> fps;
    std::optional<unsigned int> n_rec;
    std::optional<std::string> input_path;
    std::optional<std::string> output_path;
    std::optional<std::string> compute_settings_path;
    std::optional<unsigned int> start_frame;
    std::optional<unsigned int> end_frame;
    bool noskip_acc;
    bool record_raw;
    bool verbose;
    std::optional<unsigned int> frame_skip;
    std::optional<unsigned int> mp4_fps;
    bool moments_record;
};

/*! \class OptionsParser
 *
 * \brief Options parser for the command-line.
 */
class OptionsParser
{
  public:
    OptionsParser();

    virtual ~OptionsParser() = default;

    OptionsParser& operator=(const OptionsParser&) = delete;

    /*! \brief Parse the command line given by the user and fill the options descriptor.
     *
     * Will automatically call help/version print and exit.
     */
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
