/*! \file information_api.hh
 *
 * \brief Regroup all functions used for information (information display, benchmark, boundary, credits, doc).
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

class InformationApi : public IApi
{

  public:
    InformationApi(const Api* api)
        : IApi(api)
    {
    }

  private:
#pragma region Credits

    /*! \brief Authors of the project */
    static inline std::vector<std::string> authors{"Titouan Gragnic",
                                                   "Arthur Courselle",
                                                   "Gustave Herve",
                                                   "Alexis Pinson",
                                                   "Etienne Senigout",
                                                   "Bastien Gaulier",
                                                   "Simon Riou",

                                                   "Chloé Magnier",
                                                   "Noé Topeza",
                                                   "Maxime Boy-Arnould",

                                                   "Oscar Morand",
                                                   "Paul Duhot",
                                                   "Thomas Xu",
                                                   "Jules Guillou",
                                                   "Samuel Goncalves",
                                                   "Edgar Delaporte",

                                                   "Adrien Langou",
                                                   "Julien Nicolle",
                                                   "Sacha Bellier",
                                                   "David Chemaly",
                                                   "Damien Didier",

                                                   "Philippe Bernet",
                                                   "Eliott Bouhana",
                                                   "Fabien Colmagro",
                                                   "Marius Dubosc",
                                                   "Guillaume Poisson",

                                                   "Anthony Strazzella",
                                                   "Ilan Guenet",
                                                   "Nicolas Blin",
                                                   "Quentin Kaci",
                                                   "Theo Lepage",

                                                   "Loïc Bellonnet-Mottet",
                                                   "Antoine Martin",
                                                   "François Te",

                                                   "Ellena Davoine",
                                                   "Clement Fang",
                                                   "Danae Marmai",
                                                   "Hugo Verjus",

                                                   "Eloi Charpentier",
                                                   "Julien Gautier",
                                                   "Florian Lapeyre",

                                                   "Thomas Jarrossay",
                                                   "Alexandre Bartz",

                                                   "Cyril Cetre",
                                                   "Clement Ledant",

                                                   "Eric Delanghe",
                                                   "Arnaud Gaillard",
                                                   "Geoffrey Le Gourrierec",

                                                   "Jeffrey Bencteux",
                                                   "Thomas Kostas",
                                                   "Pierre Pagnoux",

                                                   "Antoine Dillée",
                                                   "Romain Cancillière",

                                                   "Michael Atlan"};

  public:
    /*! \brief Gets the credits
     *
     * \return const std::vector<std::string> credits in columns
     */
    constexpr std::vector<std::string> get_credits() const
    {
        std::vector<std::string> res{"", "", ""};

        size_t nb_columns = 3;
        for (size_t i = 0; i < authors.size(); i++)
            res[i % nb_columns] += authors[i] + "<br>";

        return res;
    }

#pragma endregion

#pragma region Benchmark

    /*! \brief Return whether the benchmark mode is activated. If activated, information about the computation (fps,
     * memory usage, etc.) will be written in a file.
     *
     * \return bool benchmark mode
     */
    inline bool get_benchmark_mode() const { return GET_SETTING(BenchmarkMode); }

    /*! \brief Activate or deactivate the benchmark mode. If activated, information about the computation (fps, memory
     * usage, etc.) will be written in a file.
     *
     * \param[in] value the new value
     */
    inline void set_benchmark_mode(bool value) const { UPDATE_SETTING(BenchmarkMode, value); }

#pragma endregion

#pragma region Information

    /*! \brief Get the boundary of the frame descriptor. It's used to choose the space transformation algorithm.
     * The formula of the boundary is: boundary = N * d^2 / lambda
     * Where:
     * N = frame height
     * d = pixel size
     * lambda = wavelength
     *
     * \return const float
     */
    float get_boundary() const;

    /*! \brief Gets the documentation url
     *
     * \return const std::string url
     */
    const std::string get_documentation_url() const;

    /*! \brief Displays information */
    void start_information_display() const;

    /*! \brief Stop the displays of information */
    void stop_information_display() const;

#pragma endregion
};

} // namespace holovibes::api