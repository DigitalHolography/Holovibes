#include "information_api.hh"

namespace holovibes::api
{

#pragma region Information

void start_information_display() { Holovibes::instance().start_information_display(); }

float get_boundary() { return Holovibes::instance().get_boundary(); }

const std::string get_documentation_url() { return "https://ftp.espci.fr/incoming/Atlan/holovibes/manual/"; }

#pragma endregion

} // namespace holovibes::api