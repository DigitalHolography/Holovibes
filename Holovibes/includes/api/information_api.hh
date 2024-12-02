/*! \file
 *
 * \brief Regroup all functions used for information (information display, benchmark, boundary, credits, doc).
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

#pragma region Credits

static std::vector<std::string> authors{"Titouan Gragnic",
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

/*! \brief Gets the credits
 *
 * \return const std::vector<std::string> credits in columns
 */
constexpr std::vector<std::string> get_credits()
{
    std::vector<std::string> res{"", "", ""};

    size_t nb_columns = 3;
    for (size_t i = 0; i < authors.size(); i++)
        res[i % nb_columns] += authors[i] + "<br>";

    return res;
}

#pragma endregion

inline bool get_benchmark_mode() { return GET_SETTING(BenchmarkMode); }
inline void set_benchmark_mode(bool value) { UPDATE_SETTING(BenchmarkMode, value); }

/*! \brief Get the boundary of frame descriptor
 *
 * \return float boundary
 */
float get_boundary();

/*! \brief Gets the documentation url
 *
 * \return const QUrl& url
 */
const QUrl get_documentation_url();

/*! \brief Displays information */
void start_information_display();

} // namespace holovibes::api