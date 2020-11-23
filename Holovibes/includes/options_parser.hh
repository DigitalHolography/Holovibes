/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

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
        std::optional<unsigned int> input_fps;
        std::optional<unsigned int> output_nb_frames;
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
}