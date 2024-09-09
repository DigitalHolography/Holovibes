/*=============================================================================
  Copyright (C) 2012 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        Frame.hpp

  Description: Inline wrapper functions for class VmbCPP::Frame.

-------------------------------------------------------------------------------

  THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF TITLE,
  NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR  PURPOSE ARE
  DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

#ifndef VMBCPP_FRAME_HPP
#define VMBCPP_FRAME_HPP

/**
* \file  Frame.hpp
*
* \brief Inline wrapper functions for class VmbCPP::Frame
*        that allocate memory for STL objects in the application's context
*        and to pass data across DLL boundaries using arrays
*/

namespace VmbCPP {

inline VmbErrorType Frame::AccessChunkData(ChunkDataAccessFunction chunkAccessFunction)
{
    VmbFrame_t* frame;
    if (VmbErrorSuccess == GetFrameStruct(frame))
    {
        return ChunkDataAccess(frame, InternalChunkDataAccessCallback, &chunkAccessFunction);
    }
    else
    {
        return static_cast<VmbErrorType>(VmbErrorInternalFault);
    }
}

inline VmbError_t Frame::InternalChunkDataAccessCallback(VmbHandle_t featureAccessHandle, void* userContext)
{    
    class ChunkFeatureContainer : public FeatureContainer
    {
    public:
        ChunkFeatureContainer(const VmbHandle_t handle)
        {
            FeatureContainer::SetHandle(handle);
        };
    };
    
    ChunkFeatureContainerPtr chunkFeatureContainer(new ChunkFeatureContainer(featureAccessHandle));

    auto chunkFunction = static_cast<ChunkDataAccessFunction*>(userContext);
    return (*chunkFunction)(chunkFeatureContainer);
}

} // namespace VmbCPP

#endif