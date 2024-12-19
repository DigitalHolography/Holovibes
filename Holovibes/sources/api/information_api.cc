#include "information_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region Information

void InformationApi::start_information_display() const { Holovibes::instance().start_information_display(); }

void InformationApi::stop_information_display() const { Holovibes::instance().stop_information_display(); }

float InformationApi::get_boundary() const
{
    camera::FrameDescriptor fd = api_->input.get_input_fd();
    const float d = api_->input.get_pixel_size() * 0.000001f;
    const float n = static_cast<float>(fd.height);
    return (n * d * d) / api_->transform.get_lambda();
}

const std::string InformationApi::get_documentation_url() const
{
    return "https://ftp.espci.fr/incoming/Atlan/holovibes/manual/";
}

#pragma endregion

} // namespace holovibes::api