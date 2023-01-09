// S640.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>

//#include "C:\Users\Vision FAE\source\repos\S640 Image Stitching\tools\tools.h"
#include <EGrabber.h>
#include <EGrabbers.h>
#include <FormatConverter.h>


int step = 0;
using namespace Euresys;
using namespace std;
namespace {

	class SubGrabber;

	class Callbacks {
	public:
		virtual void onNewBufferEvent(SubGrabber &subGrabber, size_t grabberIx, const NewBufferData &data) {};
		virtual void onIoToolboxEvent(SubGrabber &subGrabber, size_t grabberIx, const IoToolboxData &data) {};
		virtual void onCicEvent(SubGrabber &subGrabber, size_t grabberIx, const CicData &data) {};
		virtual void onDataStreamEvent(SubGrabber &subGrabber, size_t grabberIx, const DataStreamData &data) {};
		virtual void onCxpInterfaceEvent(SubGrabber &subGrabber, size_t grabberIx, const CxpInterfaceData &data) {};
		virtual void onDeviceErrorEvent(SubGrabber &subGrabber, size_t grabberIx, const DeviceErrorData &data) {};
	};

	typedef EGrabber<CallbackOnDemand> EGrabberT;

	class SubGrabber : public EGrabberT {
	public:
		SubGrabber(EGenTL &gentl, int interfaceIndex, int deviceIndex, int dataStreamIndex,
			gc::DEVICE_ACCESS_FLAGS deviceOpenFlags, bool remoteRequired)
			: EGrabberT(gentl, interfaceIndex, deviceIndex, dataStreamIndex, deviceOpenFlags, remoteRequired)
			, callbacks(0)
			, grabberIx(0)
		{}
		void setCallbacks(Callbacks *callbacks, size_t grabberIx) {
			this->callbacks = callbacks;
			this->grabberIx = grabberIx;
		}
	private:
		Callbacks *callbacks;
		size_t grabberIx;
		void onNewBufferEvent(const NewBufferData &data) {
			callbacks->onNewBufferEvent(*this, grabberIx, data);
		}
		void onIoToolboxEvent(const IoToolboxData &data) {
			callbacks->onIoToolboxEvent(*this, grabberIx, data);
		}
		void onCicEvent(const CicData &data) {
			callbacks->onCicEvent(*this, grabberIx, data);
		}
		void onDataStreamEvent(const DataStreamData &data) {
			callbacks->onDataStreamEvent(*this, grabberIx, data);
		}
		void onCxpInterfaceEvent(const CxpInterfaceData &data) {
			callbacks->onCxpInterfaceEvent(*this, grabberIx, data);
		}
		void onDeviceErrorEvent(const DeviceErrorData &data) {
			callbacks->onDeviceErrorEvent(*this, grabberIx, data);
		}
	};

	// This sample expects that all connectors of the phantom streamer 16 CXP6 are
	// properly connected to the different grabbers in the right order.
	// In addition, the sample expects that the camera configuration (number of
	// devices exposed by the camera) matches the number of available EGrabber
	// instances the system.

	class SubGrabberManager : private Callbacks {
	public:
		SubGrabberManager(EGenTL &gentl, size_t bufferCount, size_t width, size_t fullHeight, const std::string &pixelFormat, float FPS)
			: converter(gentl)
			, grabbers(gentl)
			, width(width)
			, height(0)
			, fullHeight(fullHeight)
			, pixelFormat(pixelFormat)
            , fps(FPS)
		{
			grabbers.root[0][0].reposition(0);		// master
			grabbers.root[0][1].reposition(1);		// slave1
			//grabbers.root[0][0].reposition(0);		// s2
			//grabbers.root[0][1].reposition(1);		// s3

			buffers.reserve(bufferCount);
			size_t pitch = width * gentl.imageGetBytesPerPixel(pixelFormat);
			size_t grabberCount = grabbers.length();// now set for a single Octo board 
			height = fullHeight / grabberCount;
			size_t payloadSize = pitch * fullHeight;
			//S990
			//size_t stripeHeight = 8 / grabberCount;
			//size_t stripePitch = stripeHeight * grabberCount;
			//s710
			size_t stripeHeight = 8; // for all configuration in S710
			size_t stripePitch = stripeHeight * grabberCount;
			try {
				for (size_t ix = 0; ix < grabberCount; ++ix) {
					grabbers[ix]->setCallbacks(this, ix);
					

					grabbers[ix]->setInteger<RemoteModule>("Width", static_cast<int64_t>(width));
					grabbers[ix]->setInteger<RemoteModule>("Height", static_cast<int64_t>(height));
					grabbers[ix]->setString<RemoteModule>("PixelFormat", pixelFormat);
					// configure stripes on grabber data stream
					grabbers[ix]->setString<StreamModule>("StripeArrangement", "Geometry_1X_2YM");
					grabbers[ix]->setInteger<StreamModule>("LinePitch", pitch);
					grabbers[ix]->setInteger<StreamModule>("LineWidth", pitch);
					grabbers[ix]->setInteger<StreamModule>("StripeHeight", stripeHeight);
					grabbers[ix]->setInteger<StreamModule>("StripePitch", stripePitch);
					grabbers[ix]->setInteger<StreamModule>("BlockHeight", 8); // in every config for S710
					grabbers[ix]->setInteger<StreamModule>("StripeOffset", 8 * ix);
					grabbers[ix]->setString<StreamModule>("StatisticsSamplingSelector", "LastSecond");
					grabbers[ix]->setString<StreamModule>("LUTConfiguration", "M_10x8");
				}
				while (buffers.size() < bufferCount) {
					uint8_t *base = static_cast<uint8_t *>(malloc(payloadSize));
					buffers.push_back(base);
					for (size_t ix = 0; ix < grabberCount; ++ix) {
						size_t offset = pitch * stripeHeight * ix;
						//S990
						//grabbers[ix]->announceAndQueue(UserMemory(base + offset, payloadSize - offset));
						//S640
						grabbers[ix]->announceAndQueue(UserMemory(base, payloadSize));
					}
				}
			}
			catch (...) {
				cleanup();
				throw;
			}
		}
		virtual ~SubGrabberManager() {
			cleanup();
		}
		void cleanup() {
			for (size_t ix = 0; ix < grabbers.length(); ++ix) { // set for a single octo board 
				grabbers[ix]->reallocBuffers(0);
			}
			for (size_t i = 0; i < buffers.size(); ++i) {
				free(buffers[i]);
			}
			buffers.clear();
		}
		void go(size_t bufferCount) {
			size_t grabberCount = grabbers.length(); // set for a single octo board 
			bool triggerMode = !grabbers[0]->getInteger<RemoteModule>(query::available("SyncImg"));
			if (!triggerMode) {
				// free-run mode with camera firmware older than end of September 2018
				grabbers[0]->setString<RemoteModule>("SyncImg", "External");
			}

			//****************************************************************************************************************
			//This section of the code is where parameters being set on Device Module- Frame Grabber and Remote Module Camera
			//****************************************************************************************************************
			else {
				// triggered mode needs camera firmware from end of September 2018
				grabbers[0]->setString<RemoteModule>("TriggerMode", "TriggerModeOn");   // camera in triggered mode
				grabbers[0]->setString<RemoteModule>("TriggerSource", "SWTRIGGER");     // source of trigger CXP
				//grabbers[0]->setString<RemoteModule>("AcquistionFrameRate", "200");
				grabbers[0]->setString<DeviceModule>("CameraControlMethod", "RC");      // tell grabber 0 to send trigger

				/* 100 fps -> 10000us */
				// float factor = fps / 100;
                // float cycleMinimumPeriod = 10000 / factor;
                float cycleMinimumPeriod = 1e6 / fps;
                std::string CycleMinimumPeriod = std::to_string(cycleMinimumPeriod);
				grabbers[0]->setString<DeviceModule>("CycleMinimumPeriod", CycleMinimumPeriod);  // set the trigger rate to 250K Hz
				grabbers[0]->setString<DeviceModule>("ExposureReadoutOverlap", "True"); // camera needs 2 trigger to start
				grabbers[0]->setString<DeviceModule>("ErrorSelector", "All");
				grabbers[0]->setString<RemoteModule>("TimeStamp", "TSOff");
                for (size_t ix = 0; ix < grabbers.length(); ++ix)
                {
                    grabbers[ix]->setInteger<StreamModule>("BufferPartCount", 1);
                }
			}
			Sleep(2000);
			
			// start each grabber in reverse order
			for (size_t ix = 0; ix < grabberCount; ++ix) {
                //grabbers[ix]->setInteger<StreamModule>("BufferPartCount", 100);
				grabbers[grabberCount - 1 - ix]->start(bufferCount);
			}

			if (!triggerMode) {
				// tell the camera to wait for a trigger
				Sleep(2000);
				grabbers[0]->setString<RemoteModule>("SyncImg", "Internal");
			}

			const uint64_t timeout = 100000;
			for (size_t i = 0; i < bufferCount; ++i) {
				// wait for each part of the buffer from each grabber
				for (size_t ix = 0; ix < grabberCount; ++ix) {
					grabbers[ix]->processEvent<NewBufferData>(timeout);
					std::ostringstream os;
					os << "process event " << i << " for board " << ix;
					grabbers[0]->memento(os.str());
				}
			}
			// convert & save merged buffers
			for (size_t i = 0; i < bufferCount; ++i) {
				FormatConverter::Auto rgb(converter, FormatConverter::OutputFormat("RGB8"), buffers[i], pixelFormat, width, fullHeight);
				//rgb.saveToDisk(Tools::getEnv("sample-output-path") + "/merged.NNN.tiff", i);
				rgb.saveToDisk("merged.NNNNNNN.bmp", i);
			}
		}
	private:
		void onNewBufferEvent(SubGrabber &subGrabber, size_t grabberIx, const NewBufferData &data) {
			ScopedBuffer buffer(subGrabber, data);
			std::string ser = subGrabber.getString<InterfaceModule>("SerialNumber");
			std::string dev = subGrabber.getString<DeviceModule>("DeviceID");
			uint64_t frameId = buffer.getInfo<uint64_t>(gc::BUFFER_INFO_FRAMEID);
			uint64_t time = buffer.getInfo<uint64_t>(gc::BUFFER_INFO_TIMESTAMP);
			std::string grabberSdi = grabbers[grabberIx].getSdi();
			//Tools::log("grabbers[" + Tools::toString(grabberIx) + "] (" + grabberSdi + ") got frame " + Tools::toString(frameId));
			std::ostringstream os;
			os << " grabber " << grabberIx << " got frame " << frameId;
			grabbers[0]->memento(os.str());
		};

		FormatConverter converter;
	public:
		EGrabbers<SubGrabber> grabbers;
	private:
		std::vector<uint8_t *> buffers;

		size_t width;
		size_t height;
		size_t fullHeight;
		std::string pixelFormat;
        float fps = 0;
	};

}

//*******************************************************************************************************************************'
//Main Function of the program used to set up Exposure time on Remote Module 
//********************************************************************************************************************************

int main() {
	EGenTL genTL;
	const size_t BUFFER_COUNT = 50;

	size_t width;
    size_t fullheight;
    cout << "Enter width: ";
    cin >> width;
    cout << "Enter fullheight: ";
    cin >> fullheight;

	float FrameRate;
	cout << "Enter Frame Rate: ";
    cin >> FrameRate;

	float Expvalue;

	float *Exptime;
	Exptime = &Expvalue;
	*Exptime = 9000;
	SubGrabberManager subGrabberManager(genTL, BUFFER_COUNT, width, fullheight, "Mono8", FrameRate);
	step = 1;

	/* 100 fps -> 9000us */
    float factor = FrameRate / 100;
    Expvalue = 9000 / factor;
	subGrabberManager.grabbers[0]->setFloat<RemoteModule>("ExposureTime", Expvalue);
	
	subGrabberManager.go(BUFFER_COUNT);
	subGrabberManager.grabbers[0]->setString<RemoteModule>("BalanceWhiteMarker", "BalanceWhiteMarkerOff");
	//subGrabberManager.grabbers[0]->setFloat<RemoteModule>("Gain", "2.0");
	cout << "Exposure time is: " << Expvalue <<'\n';
	
	return 0;
}

