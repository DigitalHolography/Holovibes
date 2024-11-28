#pragma once

#include "API.hh"
#include "enum_record_mode.hh"
#include "enum_recorded_data_type.hh"

namespace holovibes::api
{

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

constexpr std::vector<std::string> get_credits()
{
    std::vector<std::string> res{"", "", ""};

    size_t nb_columns = 3;
    for (size_t i = 0; i < authors.size(); i++)
        res[i % nb_columns] += authors[i] + "<br>";

    return res;
}

inline bool get_benchmark_mode() { return GET_SETTING(BenchmarkMode); }
inline void set_benchmark_mode(bool value) { UPDATE_SETTING(BenchmarkMode, value); }

/*! \name Zone
 * \{
 */
inline units::RectFd get_signal_zone() { return GET_SETTING(SignalZone); };
inline units::RectFd get_noise_zone() { return GET_SETTING(NoiseZone); };
inline units::RectFd get_composite_zone() { return GET_SETTING(CompositeZone); };
inline units::RectFd get_zoomed_zone() { return GET_SETTING(ZoomedZone); };

inline void set_signal_zone(const units::RectFd& rect) { UPDATE_SETTING(SignalZone, rect); };
inline void set_noise_zone(const units::RectFd& rect) { UPDATE_SETTING(NoiseZone, rect); };
inline void set_composite_zone(const units::RectFd& rect) { UPDATE_SETTING(CompositeZone, rect); };
inline void set_zoomed_zone(const units::RectFd& rect) { UPDATE_SETTING(ZoomedZone, rect); };

    /*! \} */

#pragma endregion

} // namespace holovibes::api
