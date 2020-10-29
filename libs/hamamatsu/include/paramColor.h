// paramCOLOR.h
// [Jan.24,2001]

#ifndef	_INCLUDE_PARAMCOLOR_H_
#define	_INCLUDE_PARAMCOLOR_H_

typedef struct {
	double	red;
	double	green;
	double	blue;
} dcam_rgbratio;

typedef struct _DCAM_PARAM_RGBRATIO
{
	DCAM_HDR_PARAM	hdr;		// id == DCAM_IDPARAM_RGBRATIO

	dcam_rgbratio	exposure;
	dcam_rgbratio	gain;
} DCAM_PARAM_RGBRATIO;

enum {
	dcamparam_rgbratio_exposure	= 0x00000001,
	dcamparam_rgbratio_gain		= 0x00000002
};

#endif
