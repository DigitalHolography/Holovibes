#pragma once

#include <optional>
#include <EGrabber.h>
#include <EGrabbers.h>

#include "camera.hh"

namespace camera
{
// using namespace Euresys;

/*! \class EHoloSubGrabber
 *
 * \brief Class to handle the different EHoloSubGrabber used to acquire images
 * from the Phantom camera with a Coaxlink Octo frame grabber.
 *
 * The events and callbacks are handled by the same thread calling get_frames
 * (camera frame read worker) as we do not need to capture multiple frames at once.
 */

using EHoloSubGrabber = Euresys::EGrabberCallbackOnDemand;

/*! \class EHoloGrabber // TODO REDO
 *
 *\brief Class to handle the different EHoloSubGrabber used to acquire images
 * from the Phantom S710 with a Coaxlink Octo frame grabber.
 *
 * This implementation supposes that the frame grabber has been configured
 * properly, through the GenICam API, so that:
 * 1. Only banks A and B are used.
 * 2. Each bank is responsible for capturing half of the full image height.
 *
 * For instance, to capture a frame of 1024*512, the first and second grabber
 * will acquire 1024*256 and stack both parts to create the full 1024*512 image.
 *
 * The documentation of the Euresys eGrabber Programmer Guide can be found at
 * https://documentation.euresys.com/Products/COAXLINK/COAXLINK_14_0/en-us/Content/00_Home/PDF_Guides.htm.
 */
class EHoloGrabberInt
{
  public:
    EHoloGrabberInt(Euresys::EGenTL& gentl,
                    unsigned int buffer_part_count,
                    std::string& pixel_format,
                    unsigned int nb_grabbers);

    virtual ~EHoloGrabberInt();

    // magic nunmber for number max of frame grabber supported (can be less for some implementation)
#define NB_MAX_GRABBER 4

    // TODO: find a better handling of the below struct
    struct SetupParam
    {
        unsigned int full_height;
        unsigned int width;
        unsigned int nb_grabbers;
        std::string pixel_format;
        size_t stripe_height;
        std::string stripe_arrangement;
        std::string& trigger_source;
        unsigned int block_height;
        unsigned int (&offsets)[NB_MAX_GRABBER];
        std::optional<std::string> trigger_mode;
        std::optional<std::string> trigger_selector;
        unsigned int cycle_minimum_period;
        float exposure_time;
        std::string& gain_selector;
        float gain;
        std::string& balance_white_marker;
        std::string flat_field_correction;
        std::string fan_ctrl;
        unsigned int acquisition_frame_rate;
    };

    virtual void setup(const SetupParam& param, Euresys::EGenTL& gentl);

    void init(unsigned int nb_buffers);

    void start();

    void stop();

    /*! \brief The width of the acquired frames. */
    unsigned int width_;

    /*! \brief The total height of the acquired frames. */
    unsigned int height_;

    /*! \brief The depth of the acquired frames. */
    PixelDepth depth_;

    /*! \brief An EGrabbers instance composed of the two EHoloSubGrabber grabbers.  */
    Euresys::EGrabbers<EHoloSubGrabber> grabbers_;

    /*! \brief The list of detected grabbers that are connected to a camera and are truly available for use. Built from
     * grabbers_ above. */
    std::vector<Euresys::EGrabberRef<EHoloSubGrabber>> available_grabbers_;

    /*! \brief Number of requested grabbers */
    unsigned int nb_grabbers_;

  private:
    /*! \brief The number of buffers used to store frames. It is equivalent to
     * the number of frames to store simultaneously.
     */
    unsigned int nb_buffers_;

    /*! \brief The number of images stored in each buffers.
     */
    unsigned int buffer_part_count_;

    /*! \brief A pointer the cuda memory allocated for the buffers.
     */
    uint8_t* ptr_;
};

class CameraPhantomInt : public Camera
{
  protected:
    CameraPhantomInt(const std::string& name, const std::string& ini_prefix);

  public:
    virtual ~CameraPhantomInt() {}

    virtual void init_camera() = 0;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual CapturedFramesDescriptor get_frames() override;

  protected:
    virtual void init_camera_(EHoloGrabberInt::SetupParam& param);

    virtual void load_ini_params() override;
    virtual void load_default_params() override;
    virtual void bind_params() override;

    std::unique_ptr<Euresys::EGenTL> gentl_;
    std::unique_ptr<EHoloGrabberInt> grabber_;

    std::string ini_prefix_;

    // TODO: maybe replace all these with an instance of SetupParam struct to simplify settings handling
    unsigned int nb_buffers_;
    unsigned int buffer_part_count_;
    unsigned int nb_grabbers_;
    unsigned int full_height_;
    unsigned int width_;

    unsigned int stripe_offsets_[NB_MAX_GRABBER];

    std::string trigger_source_;
    std::string trigger_selector_;
    float exposure_time_;
    unsigned int cycle_minimum_period_;
    std::string pixel_format_;

    std::string gain_selector_;
    std::string trigger_mode_;
    float gain_;
    std::string balance_white_marker_;
    std::string flat_field_correction_;
    std::string fan_ctrl_;
    unsigned int acquisition_frame_rate_;
};

} // namespace camera