// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_ACTIONACKSTATUSENUM_H__
#define __PV_ACTIONACKSTATUSENUM_H__

typedef enum
{
    PvActionAckStatusOK = 0,
    PvActionAckStatusLate = 1,
    PvActionAckStatusOverflow = 2,
    PvActionAckStatusNoRefTime = 3,
}
PvActionAckStatusEnum;

#endif // __PV_ACTIONACKSTATUSENUM_H__
