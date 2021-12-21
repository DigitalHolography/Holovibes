/*! \file
 *
 * \brief Contains compute parameters.
 */
#pragma once

#include <atomic>
#include <mutex>
#include "aliases.hh"
#include "observable.hh"
#include "rect.hh"

// enum
#include "enum_space_transformation.hh"
#include "enum_time_transformation.hh"
#include "enum_computation.hh"
#include "enum_access_mode.hh"
#include "enum_window_kind.hh"

// struct
#include "composite_struct.hh"
#include "view_struct.hh"

namespace holovibes
{
/*! \class ComputeDescriptor
 *
 * \brief Contains compute parameters.
 *
 * Theses parameters will be used when the pipe is refresh.
 * It defines parameters for FFT, lens (Fresnel transforms ...),
 * post-processing (contrast, shift_corners, log scale).
 *
 * The class use the *Observer* design pattern instead of the signal
 * mechanism of Qt because classes in the namespace holovibes are
 * independent of GUI or CLI implementations. So that, the code remains
 * reusable.
 *
 * This class contains std::atomic fields to avoid concurrent access between
 * the pipe and the GUI.
 */
class ComputeDescriptor : public Observable
{
  public:
    /*! \brief ComputeDescriptor constructor
     * Initialize the compute descriptor to default values of computation. */
    ComputeDescriptor();

    /*! \brief ComputeDescriptor destructor. */
    ~ComputeDescriptor();

    /*! \brief Assignment operator
     * The assignment operator is explicitely defined because std::atomic type
     * does not allow to generate assignments operator automatically.
     */
    ComputeDescriptor& operator=(const ComputeDescriptor& cd);

#pragma region Atomics vars

#pragma endregion
};
} // namespace holovibes
