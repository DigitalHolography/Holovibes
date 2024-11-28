#include "API.hh"
#include "logger.hh"
#include "notifier.hh"
#include "tools.hh"

#include <string>
#include <unordered_map>

namespace holovibes::api
{

#pragma region Information

void start_information_display() { Holovibes::instance().start_information_display(); }

float get_boundary() { return Holovibes::instance().get_boundary(); }

const QUrl get_documentation_url() { return QUrl("https://ftp.espci.fr/incoming/Atlan/holovibes/manual/"); }

#pragma endregion

} // namespace holovibes::api
