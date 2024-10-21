/*! \file camera_phantom_interface.hh
 *
 * \brief Contains base classes serving as interfaces with default implementations (common behaviour) for each camera's
 * CameraPhantom implementation
 */
#pragma once

#include <optional>
#include <EGrabber.h>
#include <EGrabbers.h>

#include "camera.hh"
#include "camera_param_map.hh"

namespace camera
{

/*! \brief Allocate a new camera and initialise it
 *
 * \return A pointer to the new camera object.
 */
template <class Cam>
Cam* InitCam();

/*! \class EHoloSubGrabber
 *
 * \brief Alias to the \ref Euresys::EGrabberCallbackOnDemand "EGrabberCallbackOnDemand" class for better code
 * lisibility
 */
using EHoloSubGrabber = Euresys::EGrabberCallbackOnDemand;

/*! \class EHoloGrabberInt
 *
 * \brief Base class serving as an interface for each camera's EHoloGrabber implementation to handle the different
 *  EHoloSubGrabber used to acquire images from an Ametek Phantom camera
 */
class EHoloGrabberInt
{
  protected:
    /*! \brief Constructor of EHoloGrabberInt. Protected to prevent instantiation outside of a derived class
     *
     * \param gentl reference to an instance of Euresys::EGenTL&
     * \param buffer_part_count buffer_part_count (i.e number of images per buffer) setting value
     * \param pixel_format reference to string specifying pixel format used by the camera
     * \param nb_grabbers specify requested number of nb_grabbers
     */
    EHoloGrabberInt(Euresys::EGenTL& gentl,
                    unsigned int buffer_part_count,
                    std::string& pixel_format,
                    unsigned int nb_grabbers);

  public:
    /*! \brief Destructor of EHoloGrabberint, free each grabber buffer and cuda memory previsouly allocated pointed by
     * ptr_ */
    virtual ~EHoloGrabberInt();

    /*! \brief Apply settings specified in the params CameraParamMap
     *
     * \param params reference to the CameraParamMap containing requested settings
     * \param gentl reference to an instance of Euresys::EGenTL&
     */
    virtual void setup(const CameraParamMap& params, Euresys::EGenTL& gentl);

    /*! \brief Allocate buffers and cuda memory
     *
     * \param nb_buffers number of buffers that was specified in the ini file
     */
    void init(unsigned int nb_buffers);

    /*! \brief Start each available grabber in reverser order */
    void start();

    /*! \brief Stop each available grabber */
    void stop();

    /*! \brief The width of the acquired frames. */
    unsigned int width_;

    /*! \brief The total height of the acquired frames. */
    unsigned int height_;

    /*! \brief The depth of the acquired frames. */
    PixelDepth depth_;

    /*! \brief An EGrabbers instance giving access to each detected frame grabber. */
    Euresys::EGrabbers<EHoloSubGrabber> grabbers_;

    /*! \brief The list of detected grabbers that are connected to a camera and are truly available for use. */
    std::vector<Euresys::EGrabberRef<EHoloSubGrabber>> available_grabbers_;

    /*! \brief Number of requested grabbers to use */
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

/*! \class CameraPhantomInt
 *
 * \brief Base class serving as interface for each camera's CameraPhantom implementation.
 */
class CameraPhantomInt : public Camera
{
  protected:
    /*! \brief Constructor of CameraPhantomInt. Protected to prevent instiation outside of a derived class
     *
     * \param ini_filename filename of the ini file associated to the camera
     * \param ini_prefix prefix of the camera for the ini file
     */
    CameraPhantomInt(const std::string& ini_filename, const std::string& ini_prefix);

  public:
    /*! \brief Destructor of CameraPhantomInt. Nothing to do */
    virtual ~CameraPhantomInt() {}

    /*! \brief Virtual pure function for camera initialization. Must be implemented in each CameraPhantom */
    virtual void init_camera() = 0;
    /*! \brief Start camera acquisition */
    virtual void start_acquisition() override;
    /*! \brief Stop camera acquisition */
    virtual void stop_acquisition() override;
    /*! \brief Shutdown camera */
    virtual void shutdown_camera() override;
    /*! \brief Handles the frame reconstitution (stiching) using each frame grabber and return them
     *
     * \return CapturedFramesDescriptor containing pointer to frames
     */
    virtual CapturedFramesDescriptor get_frames() override;

  protected:
    /*! \brief Load parameters from the INI file and store them (into private attributes).
     *
     * Reads the file stream opened to the INI file, and fill the parser object with corresponding data.
     */
    virtual void load_ini_params() override;
    /*! \brief Load default parameters for the camera.
     *
     * Fill default values in class fields and frame descriptor
     * (e.g. exposure time, ROI, fps). Each camera model has specific
     * capabilities, which is why further classes inherit from Camera to
     * implement their behaviours with appropriate their SDK.
     *
     * The camera must work with these default, fallback settings.
     */
    virtual void load_default_params() override;
    /*! \brief Set parameters with data loaded with load_ini_params().
     *
     * This method shall use the camera's API to properly modify its
     * configuration. Validity checking for any parameter is enclosed in this
     * method.
     */
    virtual void bind_params() override;

    /*! \brief Unique pointer to \ref Euresys::EGenTL "EGenTL" instance used by the camera */
    std::unique_ptr<Euresys::EGenTL> gentl_;
    /*! \brief Unique pointer to an implementation of \ref camera::EHoloGrabberInt "EHoloGrabberInt" that was
     * instantiated by the camera*/
    std::unique_ptr<EHoloGrabberInt> grabber_;

    /*! \brief Camera prefix used in the ini file */
    std::string ini_prefix_;

    /*! \brief Instance of \ref camera::CameraParamMap "CameraParamMap" containing the requested settings */
    CameraParamMap params_;

    /*! \brief Number of buffers used by the camera */
    unsigned int nb_buffers_;
    /*! \brief Height of a frame after stiching */
    unsigned int full_height_;
    /*! \brief Width of a frame */
    unsigned int width_;
};

} // namespace camera

#include "camera_phantom_interface.hxx"