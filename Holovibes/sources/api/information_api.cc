#include "information_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region Information

void InformationApi::start_information_display() const { Holovibes::instance().start_information_display(); }

float InformationApi::get_boundary() const { return Holovibes::instance().get_boundary(); }

const std::string InformationApi::get_documentation_url() const
{
    return "https://ftp.espci.fr/incoming/Atlan/holovibes/manual/";
}

#pragma endregion

} // namespace holovibes::api