
// Windows C Interface
#if !defined(__atmcd32d_h)
#define __atmcd32d_h
#pragma hdrstop

#include "windows.h"

#ifdef __cplusplus
extern "C" {
#endif

#define at_32 long
#define at_u32 unsigned long

#if defined(__BORLANDC__) && (__BORLANDC__<=0x540)  
  #define at_64 __int64
  #define at_u64 unsigned __int64
#else
  #define at_64 long long
  #define at_u64 unsigned long long
#endif
		
// ===================================
// Version Information Definitions
// ===================================

//Version Information Enumeration - NOTE: Increment the count constant below when
//                                        this structure is extended
// Using large numbers to force size to an integer
typedef enum {
	AT_SDKVersion = 0x40000000, 
	AT_DeviceDriverVersion = 0x40000001
} AT_VersionInfoId;

// Count of the number of elements in the Version Information Enumeration
// NOTE: Increment when extending enumeration
#define AT_NoOfVersionInfoIds 2

// Minimum recommended length of the Version Info buffer parameter
#define AT_VERSION_INFO_LEN 80
// Minimum recommended length of the Controller Card Model buffer parameter
#define AT_CONTROLLER_CARD_MODEL_LEN 80
// ===================================
		
// ===================================
// DDG Lite Definitions
// ===================================

//Channel enumeration
typedef enum {
// Using large numbers to force size to an integer
  AT_DDGLite_ChannelA = 0x40000000, 
	AT_DDGLite_ChannelB = 0x40000001, 
	AT_DDGLite_ChannelC = 0x40000002
} AT_DDGLiteChannelId;
// Control byte flags
#define AT_DDGLite_ControlBit_GlobalEnable   0x01

#define AT_DDGLite_ControlBit_ChannelEnable  0x01
#define AT_DDGLite_ControlBit_FreeRun        0x02
#define AT_DDGLite_ControlBit_DisableOnFrame 0x04
#define AT_DDGLite_ControlBit_RestartOnFire  0x08
#define AT_DDGLite_ControlBit_Invert         0x10
#define AT_DDGLite_ControlBit_EnableOnFire   0x20
// ===================================

// ===================================
// USB iStar Definitions
// ===================================

#define AT_DDG_POLARITY_POSITIVE  0
#define AT_DDG_POLARITY_NEGATIVE  1
#define AT_DDG_TERMINATION_50OHMS 0
#define AT_DDG_TERMINATION_HIGHZ  1

#define AT_STEPMODE_CONSTANT      0
#define AT_STEPMODE_EXPONENTIAL   1
#define AT_STEPMODE_LOGARITHMIC   2
#define AT_STEPMODE_LINEAR        3
#define AT_STEPMODE_OFF           100

#define AT_GATEMODE_FIRE_AND_GATE 0
#define AT_GATEMODE_FIRE_ONLY     1
#define AT_GATEMODE_GATE_ONLY     2
#define AT_GATEMODE_CW_ON         3
#define AT_GATEMODE_CW_OFF        4
#define AT_GATEMODE_DDG           5
// ===================================
	
typedef struct ANDORCAPS
{
  ULONG ulSize;
	ULONG ulAcqModes;
	ULONG ulReadModes;
	ULONG ulTriggerModes;
	ULONG ulCameraType;
	ULONG ulPixelMode;
	ULONG ulSetFunctions;
	ULONG ulGetFunctions;
	ULONG ulFeatures;
	ULONG ulPCICard;
	ULONG ulEMGainCapability;
	ULONG ulFTReadModes;
} AndorCapabilities;
	
typedef struct COLORDEMOSAICINFO
{
  int iX;
	int iY;
	int iAlgorithm;
	int iXPhase;
	int iYPhase;
	int iBackground;
} ColorDemosaicInfo;
	
typedef struct WHITEBALANCEINFO
{
  int iSize;
	int iX;
	int iY;
	int iAlgorithm;
	int iROI_left;
	int iROI_right;
	int iROI_top;
	int iROI_bottom;
	int iOperation;
} WhiteBalanceInfo;
	
unsigned int WINAPI AbortAcquisition(void);
unsigned int WINAPI CancelWait(void);
unsigned int WINAPI CoolerOFF(void);
unsigned int WINAPI CoolerON(void);
unsigned int WINAPI DemosaicImage(WORD * grey, WORD * red, WORD * green, WORD * blue, ColorDemosaicInfo * info);
unsigned int WINAPI EnableKeepCleans(int iMode);
unsigned int WINAPI FreeInternalMemory(void);
unsigned int WINAPI GetAcquiredData(at_32 * arr, unsigned long size);
unsigned int WINAPI GetAcquiredData16(WORD * arr, unsigned long size);
unsigned int WINAPI GetAcquiredFloatData(float * arr, unsigned long size);
unsigned int WINAPI GetAcquisitionProgress(long * acc, long * series);
unsigned int WINAPI GetAcquisitionTimings(float * exposure, float * accumulate, float * kinetic);
unsigned int WINAPI GetAdjustedRingExposureTimes(int inumTimes, float * fptimes);
unsigned int WINAPI GetAllDMAData(at_32 * arr, unsigned long size);
unsigned int WINAPI GetAmpDesc(int index, char * name, int length);
unsigned int WINAPI GetAmpMaxSpeed(int index, float * speed);
unsigned int WINAPI GetAvailableCameras(long * totalCameras);
unsigned int WINAPI GetBackground(at_32 * arr, unsigned long size);
unsigned int WINAPI GetBaselineClamp(int * state);
unsigned int WINAPI GetBitDepth(int channel, int * depth);
unsigned int WINAPI GetCameraEventStatus(DWORD * camStatus);
unsigned int WINAPI GetCameraHandle(long cameraIndex, long * cameraHandle);
unsigned int WINAPI GetCameraInformation(int index, long * information);
unsigned int WINAPI GetCameraSerialNumber(int * number);
unsigned int WINAPI GetCapabilities(AndorCapabilities * caps);
unsigned int WINAPI GetControllerCardModel(char * controllerCardModel);
unsigned int WINAPI GetCountConvertWavelengthRange(float * minval, float * maxval);
unsigned int WINAPI GetCurrentCamera(long * cameraHandle);
unsigned int WINAPI GetCYMGShift(int * iXshift, int * iYShift);
unsigned int WINAPI GetDDGExternalOutputEnabled(at_u32 uiIndex, at_u32 * puiEnabled);
unsigned int WINAPI GetDDGExternalOutputPolarity(at_u32 uiIndex, at_u32 * puiPolarity);
unsigned int WINAPI GetDDGExternalOutputStepEnabled(at_u32 uiIndex, at_u32 * puiEnabled);
unsigned int WINAPI GetDDGExternalOutputTime(at_u32 uiIndex, at_u64 * puiDelay, at_u64 * puiWidth);
unsigned int WINAPI GetDDGTTLGateWidth(at_u64 opticalWidth, at_u64 * ttlWidth);
unsigned int WINAPI GetDDGGateTime(at_u64 * puiDelay, at_u64 * puiWidth);
unsigned int WINAPI GetDDGInsertionDelay(int * piState);
unsigned int WINAPI GetDDGIntelligate(int * piState);
unsigned int WINAPI GetDDGIOC(int * state);
unsigned int WINAPI GetDDGIOCFrequency(double * frequency);
unsigned int WINAPI GetDDGIOCNumber(unsigned long * numberPulses);
unsigned int WINAPI GetDDGIOCNumberRequested(at_u32 * pulses);
unsigned int WINAPI GetDDGIOCPeriod(at_u64 * period);
unsigned int WINAPI GetDDGIOCPulses(int * pulses);
unsigned int WINAPI GetDDGIOCTrigger(at_u32 * trigger);
unsigned int WINAPI GetDDGOpticalWidthEnabled(at_u32 * puiEnabled);

// DDG Lite functions
unsigned int WINAPI GetDDGLiteGlobalControlByte(unsigned char * control);
unsigned int WINAPI GetDDGLiteControlByte(AT_DDGLiteChannelId channel, unsigned char * control);
unsigned int WINAPI GetDDGLiteInitialDelay(AT_DDGLiteChannelId channel, float * fDelay);
unsigned int WINAPI GetDDGLitePulseWidth(AT_DDGLiteChannelId channel, float * fWidth);
unsigned int WINAPI GetDDGLiteInterPulseDelay(AT_DDGLiteChannelId channel, float * fDelay);
unsigned int WINAPI GetDDGLitePulsesPerExposure(AT_DDGLiteChannelId channel, at_u32 * ui32Pulses);

unsigned int WINAPI GetDDGPulse(double wid, double resolution, double * Delay, double * Width);
unsigned int WINAPI GetDDGStepCoefficients(at_u32 mode, double * p1, double * p2);
unsigned int WINAPI GetDDGWidthStepCoefficients(at_u32 mode, double * p1, double * p2);
unsigned int WINAPI GetDDGStepMode(at_u32 * mode);
unsigned int WINAPI GetDDGWidthStepMode(at_u32 * mode);
unsigned int WINAPI GetDetector(int * xpixels, int * ypixels);
unsigned int WINAPI GetDICameraInfo(void * info);
unsigned int WINAPI GetEMAdvanced(int * state);
unsigned int WINAPI GetEMCCDGain(int * gain);
unsigned int WINAPI GetEMGainRange(int * low, int * high);
unsigned int WINAPI GetExternalTriggerTermination(at_u32 * puiTermination);
unsigned int WINAPI GetFastestRecommendedVSSpeed(int * index, float * speed);
unsigned int WINAPI GetFIFOUsage(int * FIFOusage);
unsigned int WINAPI GetFilterMode(int * mode);
unsigned int WINAPI GetFKExposureTime(float * time);
unsigned int WINAPI GetFKVShiftSpeed(int index, int * speed);
unsigned int WINAPI GetFKVShiftSpeedF(int index, float * speed);
unsigned int WINAPI GetFrontEndStatus(int * piFlag);
unsigned int WINAPI GetGateMode(int * piGatemode);
unsigned int WINAPI GetHardwareVersion(unsigned int * PCB, unsigned int * Decode, unsigned int * dummy1, unsigned int * dummy2, unsigned int * CameraFirmwareVersion, unsigned int * CameraFirmwareBuild);
unsigned int WINAPI GetHeadModel(char * name);
unsigned int WINAPI GetHorizontalSpeed(int index, int * speed);
unsigned int WINAPI GetHSSpeed(int channel, int typ, int index, float * speed);
unsigned int WINAPI GetHVflag(int * bFlag);
unsigned int WINAPI GetID(int devNum, int * id);
unsigned int WINAPI GetImageFlip(int * iHFlip, int * iVFlip);
unsigned int WINAPI GetImageRotate(int * iRotate);
unsigned int WINAPI GetImages(long first, long last, at_32 * arr, unsigned long size, long * validfirst, long * validlast);
unsigned int WINAPI GetImages16(long first, long last, WORD * arr, unsigned long size, long * validfirst, long * validlast);
unsigned int WINAPI GetImagesPerDMA(unsigned long * images);
unsigned int WINAPI GetIRQ(int * IRQ);
unsigned int WINAPI GetKeepCleanTime(float * KeepCleanTime);
unsigned int WINAPI GetMaximumBinning(int ReadMode, int HorzVert, int * MaxBinning);
unsigned int WINAPI GetMaximumExposure(float * MaxExp);
unsigned int WINAPI GetMaximumNumberRingExposureTimes(int * number);
unsigned int WINAPI GetMCPGain(int * piGain);
unsigned int WINAPI GetMCPGainRange(int * iLow, int * iHigh);
unsigned int WINAPI GetMCPGainTable(int iNum, int * piGain, float * pfPhotoepc);
unsigned int WINAPI GetMCPVoltage(int * iVoltage);
unsigned int WINAPI GetMinimumImageLength(int * MinImageLength);
unsigned int WINAPI GetMinimumNumberInSeries(int * number);
unsigned int WINAPI GetMostRecentColorImage16(unsigned long size, int algorithm, WORD * red, WORD * green, WORD * blue);
unsigned int WINAPI GetMostRecentImage(at_32 * arr, unsigned long size);
unsigned int WINAPI GetMostRecentImage16(WORD * arr, unsigned long size);
unsigned int WINAPI GetMSTimingsData(SYSTEMTIME * TimeOfStart, float * pfDifferences, int inoOfImages);
unsigned int WINAPI GetMetaDataInfo(SYSTEMTIME * TimeOfStart, float * pfTimeFromStart, unsigned int index);
unsigned int WINAPI GetMSTimingsEnabled(void);
unsigned int WINAPI GetNewData(at_32 * arr, unsigned long size);
unsigned int WINAPI GetNewData16(WORD * arr, unsigned long size);
unsigned int WINAPI GetNewData8(unsigned char * arr, unsigned long size);
unsigned int WINAPI GetNewFloatData(float * arr, unsigned long size);
unsigned int WINAPI GetNumberADChannels(int * channels);
unsigned int WINAPI GetNumberAmp(int * amp);
unsigned int WINAPI GetNumberAvailableImages(at_32 * first, at_32 * last);
unsigned int WINAPI GetNumberDDGExternalOutputs(at_u32 * puiCount);
unsigned int WINAPI GetNumberDevices(int * numDevs);
unsigned int WINAPI GetNumberFKVShiftSpeeds(int * number);
unsigned int WINAPI GetNumberHorizontalSpeeds(int * number);
unsigned int WINAPI GetNumberHSSpeeds(int channel, int typ, int * speeds);
unsigned int WINAPI GetNumberMissedExternalTriggers(unsigned int first, unsigned int last, WORD * arr, unsigned int size);
unsigned int WINAPI GetNumberNewImages(long * first, long * last);
unsigned int WINAPI GetNumberPhotonCountingDivisions(at_u32 * noOfDivisions);
unsigned int WINAPI GetNumberPreAmpGains(int * noGains);
unsigned int WINAPI GetNumberRingExposureTimes(int * ipnumTimes);
unsigned int WINAPI GetNumberIO(int * iNumber);
unsigned int WINAPI GetNumberVerticalSpeeds(int * number);
unsigned int WINAPI GetNumberVSAmplitudes(int * number);
unsigned int WINAPI GetNumberVSSpeeds(int * speeds);
unsigned int WINAPI GetOldestImage(at_32 * arr, unsigned long size);
unsigned int WINAPI GetOldestImage16(WORD * arr, unsigned long size);
unsigned int WINAPI GetPhosphorStatus(int * piFlag);
unsigned int WINAPI GetPhysicalDMAAddress(unsigned long * Address1, unsigned long * Address2);
unsigned int WINAPI GetPixelSize(float * xSize, float * ySize);
unsigned int WINAPI GetPreAmpGain(int index, float * gain);
unsigned int WINAPI GetPreAmpGainText(int index, char * name, int length);
unsigned int WINAPI GetDualExposureTimes(float * exposure1, float * exposure2);
unsigned int WINAPI GetQE(char * sensor, float wavelength, unsigned int mode, float * QE);
unsigned int WINAPI GetReadOutTime(float * ReadOutTime);
unsigned int WINAPI GetRegisterDump(int * mode);
unsigned int WINAPI GetRelativeImageTimes(unsigned int first, unsigned int last, at_u64 * arr, unsigned int size);
unsigned int WINAPI GetRingExposureRange(float * fpMin, float * fpMax);
unsigned int WINAPI GetSDK3Handle(int * Handle);
unsigned int WINAPI GetSensitivity(int channel, int horzShift, int amplifier, int pa, float * sensitivity);
unsigned int WINAPI GetShutterMinTimes(int * minclosingtime, int * minopeningtime);
unsigned int WINAPI GetSizeOfCircularBuffer(long * index);
unsigned int WINAPI GetSlotBusDeviceFunction(DWORD * dwslot, DWORD * dwBus, DWORD * dwDevice, DWORD * dwFunction);
unsigned int WINAPI GetSoftwareVersion(unsigned int * eprom, unsigned int * coffile, unsigned int * vxdrev, unsigned int * vxdver, unsigned int * dllrev, unsigned int * dllver);
unsigned int WINAPI GetSpoolProgress(long * index);
unsigned int WINAPI GetStartUpTime(float * time);
unsigned int WINAPI GetStatus(int * status);
unsigned int WINAPI GetTECStatus(int * piFlag);
unsigned int WINAPI GetTemperature(int * temperature);
unsigned int WINAPI GetTemperatureF(float * temperature);
unsigned int WINAPI GetTemperatureRange(int * mintemp, int * maxtemp);
unsigned int WINAPI GetTemperatureStatus(float * SensorTemp, float * TargetTemp, float * AmbientTemp, float * CoolerVolts);
unsigned int WINAPI GetTotalNumberImagesAcquired(long * index);
unsigned int WINAPI GetIODirection(int index, int * iDirection);
unsigned int WINAPI GetIOLevel(int index, int * iLevel);
unsigned int WINAPI GetVersionInfo(AT_VersionInfoId arr, char * szVersionInfo, at_u32 ui32BufferLen);
unsigned int WINAPI GetVerticalSpeed(int index, int * speed);
unsigned int WINAPI GetVirtualDMAAddress(void ** Address1, void ** Address2);
unsigned int WINAPI GetVSAmplitudeString(int index, char * text);
unsigned int WINAPI GetVSAmplitudeFromString(char * text, int * index);
unsigned int WINAPI GetVSAmplitudeValue(int index, int * value);
unsigned int WINAPI GetVSSpeed(int index, float * speed);
unsigned int WINAPI GPIBReceive(int id, short address, char * text, int size);
unsigned int WINAPI GPIBSend(int id, short address, char * text);
unsigned int WINAPI I2CBurstRead(BYTE i2cAddress, long nBytes, BYTE * data);
unsigned int WINAPI I2CBurstWrite(BYTE i2cAddress, long nBytes, BYTE * data);
unsigned int WINAPI I2CRead(BYTE deviceID, BYTE intAddress, BYTE * pdata);
unsigned int WINAPI I2CReset(void);
unsigned int WINAPI I2CWrite(BYTE deviceID, BYTE intAddress, BYTE data);
unsigned int WINAPI IdAndorDll(void);
unsigned int WINAPI InAuxPort(int port, int * state);
unsigned int WINAPI Initialize(char * dir);
unsigned int WINAPI InitializeDevice(char * dir);
unsigned int WINAPI IsAmplifierAvailable(int iamp);
unsigned int WINAPI IsCoolerOn(int * iCoolerStatus);
unsigned int WINAPI IsCountConvertModeAvailable(int mode);
unsigned int WINAPI IsInternalMechanicalShutter(int * InternalShutter);
unsigned int WINAPI IsPreAmpGainAvailable(int channel, int amplifier, int index, int pa, int * status);
unsigned int WINAPI IsReadoutFlippedByAmplifier(int iAmplifier, int * iFlipped);
unsigned int WINAPI IsTriggerModeAvailable(int iTriggerMode);
unsigned int WINAPI Merge(const at_32 * arr, long nOrder, long nPoint, long nPixel, float * coeff, long fit, long hbin, at_32 * output, float * start, float * step_Renamed);
unsigned int WINAPI OutAuxPort(int port, int state);
unsigned int WINAPI PrepareAcquisition(void);
unsigned int WINAPI SaveAsBmp(const char * path, const char * palette, long ymin, long ymax);
unsigned int WINAPI SaveAsCommentedSif(char * path, char * comment);
unsigned int WINAPI SaveAsEDF(char * szPath, int iMode);
unsigned int WINAPI SaveAsFITS(char * szFileTitle, int typ);
unsigned int WINAPI SaveAsRaw(char * szFileTitle, int typ);
unsigned int WINAPI SaveAsSif(char * path);
unsigned int WINAPI SaveAsSPC(char * path);
unsigned int WINAPI SaveAsTiff(char * path, char * palette, int position, int typ);
unsigned int WINAPI SaveAsTiffEx(char * path, char * palette, int position, int typ, int mode);
unsigned int WINAPI SaveEEPROMToFile(char * cFileName);
unsigned int WINAPI SaveToClipBoard(char * palette);
unsigned int WINAPI SelectDevice(int devNum);
unsigned int WINAPI SendSoftwareTrigger(void);
unsigned int WINAPI SetAccumulationCycleTime(float time);
unsigned int WINAPI SetAcqStatusEvent(HANDLE statusEvent);
unsigned int WINAPI SetAcquisitionMode(int mode);
unsigned int WINAPI SetAcquisitionType(int typ);
unsigned int WINAPI SetADChannel(int channel);
unsigned int WINAPI SetAdvancedTriggerModeState(int iState);
unsigned int WINAPI SetBackground(at_32 * arr, unsigned long size);
unsigned int WINAPI SetBaselineClamp(int state);
unsigned int WINAPI SetBaselineOffset(int offset);
unsigned int WINAPI SetCameraLinkMode(int mode);
unsigned int WINAPI SetCameraStatusEnable(DWORD Enable);
unsigned int WINAPI SetChargeShifting(unsigned int NumberRows, unsigned int NumberRepeats);
unsigned int WINAPI SetComplexImage(int numAreas, int * areas);
unsigned int WINAPI SetCoolerMode(int mode);
unsigned int WINAPI SetCountConvertMode(int Mode);
unsigned int WINAPI SetCountConvertWavelength(float wavelength);
unsigned int WINAPI SetCropMode(int active, int cropHeight, int reserved);
unsigned int WINAPI SetCurrentCamera(long cameraHandle);
unsigned int WINAPI SetCustomTrackHBin(int bin);
unsigned int WINAPI SetDataType(int typ);
unsigned int WINAPI SetDACOutput(int iOption, int iResolution, int iValue);
unsigned int WINAPI SetDACOutputScale(int iScale);
unsigned int WINAPI SetDDGAddress(BYTE t0, BYTE t1, BYTE t2, BYTE t3, BYTE address);
unsigned int WINAPI SetDDGExternalOutputEnabled(at_u32 uiIndex, at_u32 uiEnabled);
unsigned int WINAPI SetDDGExternalOutputPolarity(at_u32 uiIndex, at_u32 uiPolarity);
unsigned int WINAPI SetDDGExternalOutputStepEnabled(at_u32 uiIndex, at_u32 uiEnabled);
unsigned int WINAPI SetDDGExternalOutputTime(at_u32 uiIndex, at_u64 uiDelay, at_u64 uiWidth);
unsigned int WINAPI SetDDGGain(int gain);
unsigned int WINAPI SetDDGGateStep(double step_Renamed);
unsigned int WINAPI SetDDGGateTime(at_u64 uiDelay, at_u64 uiWidth);
unsigned int WINAPI SetDDGInsertionDelay(int state);
unsigned int WINAPI SetDDGIntelligate(int state);
unsigned int WINAPI SetDDGIOC(int state);
unsigned int WINAPI SetDDGIOCFrequency(double frequency);
unsigned int WINAPI SetDDGIOCNumber(unsigned long numberPulses);
unsigned int WINAPI SetDDGIOCPeriod(at_u64 period);
unsigned int WINAPI SetDDGIOCTrigger(at_u32 trigger);
unsigned int WINAPI SetDDGOpticalWidthEnabled(at_u32 uiEnabled);

// DDG Lite functions
unsigned int WINAPI SetDDGLiteGlobalControlByte(unsigned char control);
unsigned int WINAPI SetDDGLiteControlByte(AT_DDGLiteChannelId channel, unsigned char control);
unsigned int WINAPI SetDDGLiteInitialDelay(AT_DDGLiteChannelId channel, float fDelay);
unsigned int WINAPI SetDDGLitePulseWidth(AT_DDGLiteChannelId channel, float fWidth);
unsigned int WINAPI SetDDGLiteInterPulseDelay(AT_DDGLiteChannelId channel, float fDelay);
unsigned int WINAPI SetDDGLitePulsesPerExposure(AT_DDGLiteChannelId channel, at_u32 ui32Pulses);

unsigned int WINAPI SetDDGStepCoefficients(at_u32 mode, double p1, double p2);
unsigned int WINAPI SetDDGWidthStepCoefficients(at_u32 mode, double p1, double p2);
unsigned int WINAPI SetDDGStepMode(at_u32 mode);
unsigned int WINAPI SetDDGWidthStepMode(at_u32 mode);
unsigned int WINAPI SetDDGTimes(double t0, double t1, double t2);
unsigned int WINAPI SetDDGTriggerMode(int mode);
unsigned int WINAPI SetDDGVariableGateStep(int mode, double p1, double p2);
unsigned int WINAPI SetDelayGenerator(int board, short address, int typ);
unsigned int WINAPI SetDMAParameters(int MaxImagesPerDMA, float SecondsPerDMA);
unsigned int WINAPI SetDriverEvent(HANDLE driverEvent);
unsigned int WINAPI SetEMAdvanced(int state);
unsigned int WINAPI SetEMCCDGain(int gain);
unsigned int WINAPI SetEMClockCompensation(int EMClockCompensationFlag);
unsigned int WINAPI SetEMGainMode(int mode);
unsigned int WINAPI SetExposureTime(float time);
unsigned int WINAPI SetExternalTriggerTermination(at_u32 uiTermination);
unsigned int WINAPI SetFanMode(int mode);
unsigned int WINAPI SetFastExtTrigger(int mode);
unsigned int WINAPI SetFastKinetics(int exposedRows, int seriesLength, float time, int mode, int hbin, int vbin);
unsigned int WINAPI SetFastKineticsEx(int exposedRows, int seriesLength, float time, int mode, int hbin, int vbin, int offset);
unsigned int WINAPI SetFilterMode(int mode);
unsigned int WINAPI SetFilterParameters(int width, float sensitivity, int range, float accept, int smooth, int noise);
unsigned int WINAPI SetFKVShiftSpeed(int index);
unsigned int WINAPI SetFPDP(int state);
unsigned int WINAPI SetFrameTransferMode(int mode);
unsigned int WINAPI SetFrontEndEvent(HANDLE driverEvent);
unsigned int WINAPI SetFullImage(int hbin, int vbin);
unsigned int WINAPI SetFVBHBin(int bin);
unsigned int WINAPI SetGain(int gain);
unsigned int WINAPI SetGate(float delay, float width, float stepRenamed);
unsigned int WINAPI SetGateMode(int gatemode);
unsigned int WINAPI SetHighCapacity(int state);
unsigned int WINAPI SetHorizontalSpeed(int index);
unsigned int WINAPI SetHSSpeed(int typ, int index);
unsigned int WINAPI SetImage(int hbin, int vbin, int hstart, int hend, int vstart, int vend);
unsigned int WINAPI SetImageFlip(int iHFlip, int iVFlip);
unsigned int WINAPI SetImageRotate(int iRotate);
unsigned int WINAPI SetIsolatedCropMode(int active, int cropheight, int cropwidth, int vbin, int hbin);
unsigned int WINAPI SetIsolatedCropModeEx(int active, int cropheight, int cropwidth, int vbin, int hbin, int cropleft, int cropbottom);
unsigned int WINAPI SetKineticCycleTime(float time);
unsigned int WINAPI SetMCPGain(int gain);
unsigned int WINAPI SetMCPGating(int gating);
unsigned int WINAPI SetMessageWindow(HWND wnd);
unsigned int WINAPI SetMetaData(int state);
unsigned int WINAPI SetMultiTrack(int number, int height, int offset, int * bottom, int * gap);
unsigned int WINAPI SetMultiTrackHBin(int bin);
unsigned int WINAPI SetMultiTrackHRange(int iStart, int iEnd);
unsigned int WINAPI SetMultiTrackScan(int trackHeight, int numberTracks, int iSIHStart, int iSIHEnd, int trackHBinning, int trackVBinning, int trackGap, int trackOffset, int trackSkip, int numberSubFrames);
unsigned int WINAPI SetNextAddress(at_32 * data, long lowAdd, long highAdd, long length, long physical);
unsigned int WINAPI SetNextAddress16(at_32 * data, long lowAdd, long highAdd, long length, long physical);
unsigned int WINAPI SetNumberAccumulations(int number);
unsigned int WINAPI SetNumberKinetics(int number);
unsigned int WINAPI SetNumberPrescans(int iNumber);
unsigned int WINAPI SetOutputAmplifier(int typ);
unsigned int WINAPI SetOverlapMode(int mode);
unsigned int WINAPI SetPCIMode(int mode, int value);
unsigned int WINAPI SetPhotonCounting(int state);
unsigned int WINAPI SetPhotonCountingThreshold(long min, long max);
unsigned int WINAPI SetPhosphorEvent(HANDLE driverEvent);
unsigned int WINAPI SetPhotonCountingDivisions(at_u32 noOfDivisions, at_32 * divisions);
unsigned int WINAPI SetPixelMode(int bitdepth, int colormode);
unsigned int WINAPI SetPreAmpGain(int index);
unsigned int WINAPI SetDualExposureTimes(float expTime1, float expTime2);
unsigned int WINAPI SetDualExposureMode(int mode);
unsigned int WINAPI SetRandomTracks(int numTracks, int * areas);
unsigned int WINAPI SetReadMode(int mode);
unsigned int WINAPI SetReadoutRegisterPacking(unsigned int mode);
unsigned int WINAPI SetRegisterDump(int mode);
unsigned int WINAPI SetRingExposureTimes(int numTimes, float * times);
unsigned int WINAPI SetSaturationEvent(HANDLE saturationEvent);
unsigned int WINAPI SetShutter(int typ, int mode, int closingtime, int openingtime);
unsigned int WINAPI SetShutterEx(int typ, int mode, int closingtime, int openingtime, int extmode);
unsigned int WINAPI SetShutters(int typ, int mode, int closingtime, int openingtime, int exttype, int extmode, int dummy1, int dummy2);
unsigned int WINAPI SetSifComment(char * comment);
unsigned int WINAPI SetSingleTrack(int centre, int height);
unsigned int WINAPI SetSingleTrackHBin(int bin);
unsigned int WINAPI SetSpool(int active, int method, char * path, int framebuffersize);
unsigned int WINAPI SetSpoolThreadCount(int count);
unsigned int WINAPI SetStorageMode(long mode);
unsigned int WINAPI SetTECEvent(HANDLE driverEvent);
unsigned int WINAPI SetTemperature(int temperature);
unsigned int WINAPI SetTemperatureEvent(HANDLE temperatureEvent);
unsigned int WINAPI SetTriggerMode(int mode);
unsigned int WINAPI SetTriggerInvert(int mode);
unsigned int WINAPI GetTriggerLevelRange(float * minimum, float * maximum);
unsigned int WINAPI SetTriggerLevel(float f_level);
unsigned int WINAPI SetIODirection(int index, int iDirection);
unsigned int WINAPI SetIOLevel(int index, int iLevel);
unsigned int WINAPI SetUserEvent(HANDLE userEvent);
unsigned int WINAPI SetUSGenomics(long width, long height);
unsigned int WINAPI SetVerticalRowBuffer(int rows);
unsigned int WINAPI SetVerticalSpeed(int index);
unsigned int WINAPI SetVirtualChip(int state);
unsigned int WINAPI SetVSAmplitude(int index);
unsigned int WINAPI SetVSSpeed(int index);
unsigned int WINAPI ShutDown(void);
unsigned int WINAPI StartAcquisition(void);
unsigned int WINAPI UnMapPhysicalAddress(void);
unsigned int WINAPI WaitForAcquisition(void);
unsigned int WINAPI WaitForAcquisitionByHandle(long cameraHandle);
unsigned int WINAPI WaitForAcquisitionByHandleTimeOut(long cameraHandle, int iTimeOutMs);
unsigned int WINAPI WaitForAcquisitionTimeOut(int iTimeOutMs);
unsigned int WINAPI WhiteBalance(WORD * wRed, WORD * wGreen, WORD * wBlue, float * fRelR, float * fRelB, WhiteBalanceInfo * info);

unsigned int WINAPI OA_Initialize(const char * const pcFilename, unsigned int uiFileNameLen);
unsigned int WINAPI OA_EnableMode(const char * const pcModeName);
unsigned int WINAPI OA_GetModeAcqParams(const char * const pcModeName, char * const pcListOfParams);
unsigned int WINAPI OA_GetUserModeNames(char * pcListOfModes);
unsigned int WINAPI OA_GetPreSetModeNames(char * pcListOfModes);
unsigned int WINAPI OA_GetNumberOfUserModes(unsigned int * const puiNumberOfModes);
unsigned int WINAPI OA_GetNumberOfPreSetModes(unsigned int * const puiNumberOfModes);
unsigned int WINAPI OA_GetNumberOfAcqParams(const char * const pcModeName, unsigned int * const puiNumberOfParams);
unsigned int WINAPI OA_AddMode(char * pcModeName, unsigned int uiModeNameLen, char * pcModeDescription, unsigned int uiModeDescriptionLen);
unsigned int WINAPI OA_WriteToFile(const char * const pcFileName, unsigned int uiFileNameLen);
unsigned int WINAPI OA_DeleteMode(const char * const pcModeName, unsigned int uiModeNameLen);
unsigned int WINAPI OA_SetInt(const char * const pcModeName, const char * pcModeParam, const int iIntValue);
unsigned int WINAPI OA_SetFloat(const char * const pcModeName, const char * pcModeParam, const float fFloatValue);
unsigned int WINAPI OA_SetString(const char * const pcModeName, const char * pcModeParam, char * pcStringValue, const unsigned int uiStringLen);
unsigned int WINAPI OA_GetInt(const char * const pcModeName, const char * const pcModeParam, int * iIntValue);
unsigned int WINAPI OA_GetFloat(const char * const pcModeName, const char * const pcModeParam, float * fFloatValue);
unsigned int WINAPI OA_GetString(const char * const pcModeName, const char * const pcModeParam, char * pcStringValue, const unsigned int uiStringLen);

unsigned int WINAPI Filter_SetMode(unsigned int mode);
unsigned int WINAPI Filter_GetMode(unsigned int * mode);
unsigned int WINAPI Filter_SetThreshold(float threshold);
unsigned int WINAPI Filter_GetThreshold(float * threshold);
unsigned int WINAPI Filter_SetDataAveragingMode(int mode);
unsigned int WINAPI Filter_GetDataAveragingMode(int * mode);
unsigned int WINAPI Filter_SetAveragingFrameCount(int frames);
unsigned int WINAPI Filter_GetAveragingFrameCount(int * frames);
unsigned int WINAPI Filter_SetAveragingFactor(int averagingFactor);
unsigned int WINAPI Filter_GetAveragingFactor(int * averagingFactor);

unsigned int WINAPI PostProcessNoiseFilter(at_32 * pInputImage, at_32 * pOutputImage, int iOutputBufferSize, int iBaseline, int iMode, float fThreshold, int iHeight, int iWidth);
unsigned int WINAPI PostProcessCountConvert(at_32 * pInputImage, at_32 * pOutputImage, int iOutputBufferSize, int iNumImages, int iBaseline, int iMode, int iEmGain, float fQE, float fSensitivity, int iHeight, int iWidth);
unsigned int WINAPI PostProcessPhotonCounting(at_32 * pInputImage, at_32 * pOutputImage, int iOutputBufferSize, int iNumImages, int iNumframes, int iNumberOfThresholds, float * pfThreshold, int iHeight, int iWidth);
unsigned int WINAPI PostProcessDataAveraging(at_32 * pInputImage, at_32 * pOutputImage, int iOutputBufferSize, int iNumImages, int iAveragingFilterMode, int iHeight, int iWidth, int iFrameCount, int iAveragingFactor);

#define DRV_ERROR_CODES 20001
#define DRV_SUCCESS 20002
#define DRV_VXDNOTINSTALLED 20003
#define DRV_ERROR_SCAN 20004
#define DRV_ERROR_CHECK_SUM 20005
#define DRV_ERROR_FILELOAD 20006
#define DRV_UNKNOWN_FUNCTION 20007
#define DRV_ERROR_VXD_INIT 20008
#define DRV_ERROR_ADDRESS 20009
#define DRV_ERROR_PAGELOCK 20010
#define DRV_ERROR_PAGEUNLOCK 20011
#define DRV_ERROR_BOARDTEST 20012
#define DRV_ERROR_ACK 20013
#define DRV_ERROR_UP_FIFO 20014
#define DRV_ERROR_PATTERN 20015

#define DRV_ACQUISITION_ERRORS 20017
#define DRV_ACQ_BUFFER 20018
#define DRV_ACQ_DOWNFIFO_FULL 20019
#define DRV_PROC_UNKONWN_INSTRUCTION 20020
#define DRV_ILLEGAL_OP_CODE 20021
#define DRV_KINETIC_TIME_NOT_MET 20022
#define DRV_ACCUM_TIME_NOT_MET 20023
#define DRV_NO_NEW_DATA 20024
#define DRV_PCI_DMA_FAIL 20025
#define DRV_SPOOLERROR 20026
#define DRV_SPOOLSETUPERROR 20027
#define DRV_FILESIZELIMITERROR 20028
#define DRV_ERROR_FILESAVE 20029

#define DRV_TEMPERATURE_CODES 20033
#define DRV_TEMPERATURE_OFF 20034
#define DRV_TEMPERATURE_NOT_STABILIZED 20035
#define DRV_TEMPERATURE_STABILIZED 20036
#define DRV_TEMPERATURE_NOT_REACHED 20037
#define DRV_TEMPERATURE_OUT_RANGE 20038
#define DRV_TEMPERATURE_NOT_SUPPORTED 20039
#define DRV_TEMPERATURE_DRIFT 20040

#define DRV_TEMP_CODES 20033
#define DRV_TEMP_OFF 20034
#define DRV_TEMP_NOT_STABILIZED 20035
#define DRV_TEMP_STABILIZED 20036
#define DRV_TEMP_NOT_REACHED 20037
#define DRV_TEMP_OUT_RANGE 20038
#define DRV_TEMP_NOT_SUPPORTED 20039
#define DRV_TEMP_DRIFT 20040

#define DRV_GENERAL_ERRORS 20049
#define DRV_INVALID_AUX 20050
#define DRV_COF_NOTLOADED 20051
#define DRV_FPGAPROG 20052
#define DRV_FLEXERROR 20053
#define DRV_GPIBERROR 20054
#define DRV_EEPROMVERSIONERROR 20055

#define DRV_DATATYPE 20064
#define DRV_DRIVER_ERRORS 20065
#define DRV_P1INVALID 20066
#define DRV_P2INVALID 20067
#define DRV_P3INVALID 20068
#define DRV_P4INVALID 20069
#define DRV_INIERROR 20070
#define DRV_COFERROR 20071
#define DRV_ACQUIRING 20072
#define DRV_IDLE 20073
#define DRV_TEMPCYCLE 20074
#define DRV_NOT_INITIALIZED 20075
#define DRV_P5INVALID 20076
#define DRV_P6INVALID 20077
#define DRV_INVALID_MODE 20078
#define DRV_INVALID_FILTER 20079

#define DRV_I2CERRORS 20080
#define DRV_I2CDEVNOTFOUND 20081
#define DRV_I2CTIMEOUT 20082
#define DRV_P7INVALID 20083
#define DRV_P8INVALID 20084
#define DRV_P9INVALID 20085
#define DRV_P10INVALID 20086
#define DRV_P11INVALID 20087

#define DRV_USBERROR 20089
#define DRV_IOCERROR 20090
#define DRV_VRMVERSIONERROR 20091
#define DRV_GATESTEPERROR 20092
#define DRV_USB_INTERRUPT_ENDPOINT_ERROR 20093
#define DRV_RANDOM_TRACK_ERROR 20094
#define DRV_INVALID_TRIGGER_MODE 20095
#define DRV_LOAD_FIRMWARE_ERROR 20096
#define DRV_DIVIDE_BY_ZERO_ERROR 20097
#define DRV_INVALID_RINGEXPOSURES 20098
#define DRV_BINNING_ERROR 20099
#define DRV_INVALID_AMPLIFIER 20100
#define DRV_INVALID_COUNTCONVERT_MODE 20101

#define DRV_ERROR_NOCAMERA 20990
#define DRV_NOT_SUPPORTED 20991
#define DRV_NOT_AVAILABLE 20992

#define DRV_ERROR_MAP 20115
#define DRV_ERROR_UNMAP 20116
#define DRV_ERROR_MDL 20117
#define DRV_ERROR_UNMDL 20118
#define DRV_ERROR_BUFFSIZE 20119
#define DRV_ERROR_NOHANDLE 20121

#define DRV_GATING_NOT_AVAILABLE 20130
#define DRV_FPGA_VOLTAGE_ERROR 20131

#define DRV_OW_CMD_FAIL 20150
#define DRV_OWMEMORY_BAD_ADDR 20151
#define DRV_OWCMD_NOT_AVAILABLE 20152
#define DRV_OW_NO_SLAVES 20153
#define DRV_OW_NOT_INITIALIZED 20154
#define DRV_OW_ERROR_SLAVE_NUM 20155
#define DRV_MSTIMINGS_ERROR 20156

#define DRV_OA_NULL_ERROR 20173
#define DRV_OA_PARSE_DTD_ERROR 20174
#define DRV_OA_DTD_VALIDATE_ERROR 20175
#define DRV_OA_FILE_ACCESS_ERROR 20176
#define DRV_OA_FILE_DOES_NOT_EXIST 20177
#define DRV_OA_XML_INVALID_OR_NOT_FOUND_ERROR 20178
#define DRV_OA_PRESET_FILE_NOT_LOADED 20179
#define DRV_OA_USER_FILE_NOT_LOADED 20180
#define DRV_OA_PRESET_AND_USER_FILE_NOT_LOADED 20181
#define DRV_OA_INVALID_FILE 20182
#define DRV_OA_FILE_HAS_BEEN_MODIFIED 20183
#define DRV_OA_BUFFER_FULL 20184
#define DRV_OA_INVALID_STRING_LENGTH 20185
#define DRV_OA_INVALID_CHARS_IN_NAME 20186
#define DRV_OA_INVALID_NAMING 20187
#define DRV_OA_GET_CAMERA_ERROR 20188
#define DRV_OA_MODE_ALREADY_EXISTS 20189
#define DRV_OA_STRINGS_NOT_EQUAL 20190
#define DRV_OA_NO_USER_DATA 20191
#define DRV_OA_VALUE_NOT_SUPPORTED 20192
#define DRV_OA_MODE_DOES_NOT_EXIST 20193
#define DRV_OA_CAMERA_NOT_SUPPORTED 20194
#define DRV_OA_FAILED_TO_GET_MODE 20195

#define DRV_PROCESSING_FAILED 20211

#define AC_ACQMODE_SINGLE 1
#define AC_ACQMODE_VIDEO 2
#define AC_ACQMODE_ACCUMULATE 4
#define AC_ACQMODE_KINETIC 8
#define AC_ACQMODE_FRAMETRANSFER 16
#define AC_ACQMODE_FASTKINETICS 32
#define AC_ACQMODE_OVERLAP 64

#define AC_READMODE_FULLIMAGE 1
#define AC_READMODE_SUBIMAGE 2
#define AC_READMODE_SINGLETRACK 4
#define AC_READMODE_FVB 8
#define AC_READMODE_MULTITRACK 16
#define AC_READMODE_RANDOMTRACK 32
#define AC_READMODE_MULTITRACKSCAN 64

#define AC_TRIGGERMODE_INTERNAL 1
#define AC_TRIGGERMODE_EXTERNAL 2
#define AC_TRIGGERMODE_EXTERNAL_FVB_EM 4
#define AC_TRIGGERMODE_CONTINUOUS 8
#define AC_TRIGGERMODE_EXTERNALSTART 16
#define AC_TRIGGERMODE_EXTERNALEXPOSURE 32
#define AC_TRIGGERMODE_INVERTED 0x40
#define AC_TRIGGERMODE_EXTERNAL_CHARGESHIFTING 0x80

// Deprecated for AC_TRIGGERMODE_EXTERNALEXPOSURE
#define AC_TRIGGERMODE_BULB 32

#define AC_CAMERATYPE_PDA 0
#define AC_CAMERATYPE_IXON 1
#define AC_CAMERATYPE_ICCD 2
#define AC_CAMERATYPE_EMCCD 3
#define AC_CAMERATYPE_CCD 4
#define AC_CAMERATYPE_ISTAR 5
#define AC_CAMERATYPE_VIDEO 6
#define AC_CAMERATYPE_IDUS 7
#define AC_CAMERATYPE_NEWTON 8
#define AC_CAMERATYPE_SURCAM 9
#define AC_CAMERATYPE_USBICCD 10
#define AC_CAMERATYPE_LUCA 11
#define AC_CAMERATYPE_RESERVED 12
#define AC_CAMERATYPE_IKON 13
#define AC_CAMERATYPE_INGAAS 14
#define AC_CAMERATYPE_IVAC 15
#define AC_CAMERATYPE_UNPROGRAMMED 16
#define AC_CAMERATYPE_CLARA 17
#define AC_CAMERATYPE_USBISTAR 18
#define AC_CAMERATYPE_SIMCAM 19
#define AC_CAMERATYPE_NEO 20
#define AC_CAMERATYPE_IXONULTRA 21
#define AC_CAMERATYPE_VOLMOS 22
#define AC_CAMERATYPE_IVACULTRA 23

#define AC_PIXELMODE_8BIT 1
#define AC_PIXELMODE_14BIT 2
#define AC_PIXELMODE_16BIT 4
#define AC_PIXELMODE_32BIT 8

#define AC_PIXELMODE_MONO 0x000000
#define AC_PIXELMODE_RGB 0x010000
#define AC_PIXELMODE_CMY 0x020000

#define AC_SETFUNCTION_VREADOUT 0x01
#define AC_SETFUNCTION_HREADOUT 0x02
#define AC_SETFUNCTION_TEMPERATURE 0x04
#define AC_SETFUNCTION_MCPGAIN 0x08
#define AC_SETFUNCTION_EMCCDGAIN 0x10
#define AC_SETFUNCTION_BASELINECLAMP 0x20
#define AC_SETFUNCTION_VSAMPLITUDE 0x40
#define AC_SETFUNCTION_HIGHCAPACITY 0x80
#define AC_SETFUNCTION_BASELINEOFFSET 0x0100
#define AC_SETFUNCTION_PREAMPGAIN 0x0200
#define AC_SETFUNCTION_CROPMODE 0x0400
#define AC_SETFUNCTION_DMAPARAMETERS 0x0800
#define AC_SETFUNCTION_HORIZONTALBIN 0x1000
#define AC_SETFUNCTION_MULTITRACKHRANGE 0x2000
#define AC_SETFUNCTION_RANDOMTRACKNOGAPS 0x4000
#define AC_SETFUNCTION_EMADVANCED 0x8000
#define AC_SETFUNCTION_GATEMODE 0x010000
#define AC_SETFUNCTION_DDGTIMES 0x020000
#define AC_SETFUNCTION_IOC 0x040000
#define AC_SETFUNCTION_INTELLIGATE 0x080000
#define AC_SETFUNCTION_INSERTION_DELAY 0x100000
#define AC_SETFUNCTION_GATESTEP 0x200000
#define AC_SETFUNCTION_GATEDELAYSTEP 0x200000
#define AC_SETFUNCTION_TRIGGERTERMINATION 0x400000
#define AC_SETFUNCTION_EXTENDEDNIR 0x800000
#define AC_SETFUNCTION_SPOOLTHREADCOUNT 0x1000000
#define AC_SETFUNCTION_REGISTERPACK 0x2000000
#define AC_SETFUNCTION_PRESCANS 0x4000000
#define AC_SETFUNCTION_GATEWIDTHSTEP 0x8000000
#define AC_SETFUNCTION_EXTENDED_CROP_MODE 0x10000000

// Deprecated for AC_SETFUNCTION_MCPGAIN
#define AC_SETFUNCTION_GAIN 8
#define AC_SETFUNCTION_ICCDGAIN 8

#define AC_GETFUNCTION_TEMPERATURE 0x01
#define AC_GETFUNCTION_TARGETTEMPERATURE 0x02
#define AC_GETFUNCTION_TEMPERATURERANGE 0x04
#define AC_GETFUNCTION_DETECTORSIZE 0x08
#define AC_GETFUNCTION_MCPGAIN 0x10
#define AC_GETFUNCTION_EMCCDGAIN 0x20
#define AC_GETFUNCTION_HVFLAG 0x40
#define AC_GETFUNCTION_GATEMODE 0x80
#define AC_GETFUNCTION_DDGTIMES 0x0100
#define AC_GETFUNCTION_IOC 0x0200
#define AC_GETFUNCTION_INTELLIGATE 0x0400
#define AC_GETFUNCTION_INSERTION_DELAY 0x0800
#define AC_GETFUNCTION_GATESTEP 0x1000
#define AC_GETFUNCTION_GATEDELAYSTEP 0x1000
#define AC_GETFUNCTION_PHOSPHORSTATUS 0x2000
#define AC_GETFUNCTION_MCPGAINTABLE 0x4000
#define AC_GETFUNCTION_BASELINECLAMP 0x8000
#define AC_GETFUNCTION_GATEWIDTHSTEP 0x10000

// Deprecated for AC_GETFUNCTION_MCPGAIN
#define AC_GETFUNCTION_GAIN 0x10
#define AC_GETFUNCTION_ICCDGAIN 0x10

#define AC_FEATURES_POLLING 1
#define AC_FEATURES_EVENTS 2
#define AC_FEATURES_SPOOLING 4
#define AC_FEATURES_SHUTTER 8
#define AC_FEATURES_SHUTTEREX 16
#define AC_FEATURES_EXTERNAL_I2C 32
#define AC_FEATURES_SATURATIONEVENT 64
#define AC_FEATURES_FANCONTROL 128
#define AC_FEATURES_MIDFANCONTROL 256
#define AC_FEATURES_TEMPERATUREDURINGACQUISITION 512
#define AC_FEATURES_KEEPCLEANCONTROL 1024
#define AC_FEATURES_DDGLITE 0x0800
#define AC_FEATURES_FTEXTERNALEXPOSURE 0x1000
#define AC_FEATURES_KINETICEXTERNALEXPOSURE 0x2000
#define AC_FEATURES_DACCONTROL 0x4000
#define AC_FEATURES_METADATA 0x8000
#define AC_FEATURES_IOCONTROL 0x10000
#define AC_FEATURES_PHOTONCOUNTING 0x20000
#define AC_FEATURES_COUNTCONVERT 0x40000
#define AC_FEATURES_DUALMODE 0x80000
#define AC_FEATURES_OPTACQUIRE 0x100000
#define AC_FEATURES_REALTIMESPURIOUSNOISEFILTER 0x200000
#define AC_FEATURES_POSTPROCESSSPURIOUSNOISEFILTER 0x400000
#define AC_FEATURES_DUALPREAMPGAIN 0x800000
#define AC_FEATURES_DEFECT_CORRECTION 0x1000000
#define AC_FEATURES_STARTOFEXPOSURE_EVENT 0x2000000
#define AC_FEATURES_ENDOFEXPOSURE_EVENT 0x4000000
#define AC_FEATURES_CAMERALINK 0x8000000

#define AC_EMGAIN_8BIT 1
#define AC_EMGAIN_12BIT 2
#define AC_EMGAIN_LINEAR12 4
#define AC_EMGAIN_REAL12 8


#ifdef __cplusplus
}
#endif

#endif
