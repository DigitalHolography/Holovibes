/*! \file API.hh
 *
 * \brief This file contains the API functions for the Holovibes application. These functions manage input files,
 * camera operations, computation settings, visualization modes, and more. The API functions are used to interact with
 * the Holovibes application from the user interface.
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

class Api
{

  private:
    // Private ctor
    Api()
    {
        composite.set_api(this);
        compute.set_api(this);
        contrast.set_api(this);
        filter2d.set_api(this);
        global_pp.set_api(this);
        information.set_api(this);
        input.set_api(this);
        record.set_api(this);
        transform.set_api(this);
        view.set_api(this);
        window_pp.set_api(this);
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
    CompositeApi composite;
    ComputeApi compute;
    ContrastApi contrast;
    Filter2dApi filter2d;
    GlobalPostProcessApi global_pp;
    InformationApi information;
    InputApi input;
    RecordApi record;
    TransformApi transform;
    ViewApi view;
    WindowPostProcessApi window_pp;
    ComputeSettingsApi settings;
};

} // namespace holovibes::api