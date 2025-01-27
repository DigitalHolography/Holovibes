/*! \file API.hh
 *
 * \brief This file contains the API functions for the Holovibes application. These functions manage input files,
 * camera operations, computation settings, visualization modes, and more. The API functions are used to interact with
 * the Holovibes application.
 */

#pragma once

#include "common_api.hh"
#include "composite_api.hh"
#include "record_api.hh"
#include "input_api.hh"
#include "view_api.hh"
#include "filter2d_api.hh"
#include "globalpostprocess_api.hh"
#include "windowpostprocess_api.hh"
#include "contrast_api.hh"
#include "compute_api.hh"
#include "transform_api.hh"
#include "information_api.hh"

#include "compute_settings.hh"

namespace holovibes::api
{

#define API holovibes::api::Api::instance()

/*! \class Api
 *
 * \brief Regroup all functions used to interact with the Holovibes application. It's a singleton, you must not call
 * functions outside of the API. The API is divided into several sub-APIs, each of which is responsible for a specific
 * aspect of the application (input, recording, contrast, ...).
 */
class Api
{

  private:
    // Private ctor
    Api()
        : composite(this)
        , compute(this)
        , contrast(this)
        , filter2d(this)
        , global_pp(this)
        , information(this)
        , input(this)
        , record(this)
        , transform(this)
        , view(this)
        , window_pp(this)
        , settings()
    {
    }

    Api(const Api&) = delete;
    Api& operator=(const Api&) = delete;

  public:
    // Singleton
    static Api& instance()
    {
        static Api instance;
        return instance;
    }

  public:
    const CompositeApi composite;
    const ComputeApi compute;
    const ContrastApi contrast;
    const Filter2dApi filter2d;
    const GlobalPostProcessApi global_pp;
    InformationApi information;
    const InputApi input;
    const RecordApi record;
    const TransformApi transform;
    const ViewApi view;
    const WindowPostProcessApi window_pp;
    const ComputeSettingsApi settings;
};

} // namespace holovibes::api