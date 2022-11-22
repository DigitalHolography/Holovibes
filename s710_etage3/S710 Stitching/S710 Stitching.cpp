#include "pch.h"
#include <iostream>

#include <EGrabber.h>
#include <EGrabbers.h>
#include <FormatConverter.h>

using namespace Euresys;
using namespace std;

namespace
{
	class SubGrabber : public EGrabberCallbackOnDemand
	{
	public:
		SubGrabber(EGenTL &gentl, int interfaceIndex, int deviceIndex, int dataStreamIndex,
			gc::DEVICE_ACCESS_FLAGS deviceOpenFlags, bool remoteRequired)
			: EGrabberCallbackOnDemand(gentl, interfaceIndex, deviceIndex, dataStreamIndex, deviceOpenFlags, remoteRequired)
		{}

	private:
		void onNewBufferEvent(const NewBufferData &data)
		{
			ScopedBuffer buffer(*this, data);
		}
	};

	class SubGrabberManager {
	public:
		SubGrabberManager(EGenTL &gentl, size_t bufferCount, size_t width, size_t fullHeight, const std::string &pixelFormat)
			: converter(gentl)
			, grabbers(gentl)
			, width(width)
			, height(0)
			, fullHeight(fullHeight)
			, pixelFormat(pixelFormat)
		{
			grabbers.root[0][0].reposition(0);		// master
			grabbers.root[0][1].reposition(1);		// slave1
			grabbers.root[1][0].reposition(2);		// s2
			grabbers.root[1][1].reposition(3);		// s3

			buffers.reserve(bufferCount);

			size_t pitch = width * gentl.imageGetBytesPerPixel(pixelFormat);
			size_t grabberCount = grabbers.length();
			height = fullHeight / grabberCount;
			size_t payloadSize = pitch * fullHeight;
			size_t stripeHeight = 8;
			size_t stripePitch = stripeHeight * grabberCount;

			try {
				for (size_t i = 0; i < grabberCount; ++i)
				{					
					grabbers[i]->setString<RemoteModule>("PixelFormat", pixelFormat);
					grabbers[i]->setInteger<RemoteModule>("Width", static_cast<int64_t>(width));
					grabbers[i]->setInteger<RemoteModule>("Height", static_cast<int64_t>(height));
					grabbers[i]->setString<StreamModule>("StripeArrangement", "Geometry_1X_2YM");
					grabbers[i]->setInteger<StreamModule>("LinePitch", pitch);
					grabbers[i]->setInteger<StreamModule>("LineWidth", pitch);
					grabbers[i]->setInteger<StreamModule>("StripeHeight", stripeHeight);
					grabbers[i]->setInteger<StreamModule>("StripePitch", stripePitch);
					grabbers[i]->setInteger<StreamModule>("BlockHeight", 8);
					grabbers[i]->setInteger<StreamModule>("StripeOffset", 8 * i);
					grabbers[i]->setString<StreamModule>("StatisticsSamplingSelector", "LastSecond");
					grabbers[i]->setString<StreamModule>("LUTConfiguration", "M_10x8");
				}

				while (buffers.size() < bufferCount)
				{
					uint8_t *base = static_cast<uint8_t *>(malloc(payloadSize));
					buffers.push_back(base);
					for (size_t ix = 0; ix < grabberCount; ++ix)
						grabbers[ix]->announceAndQueue(UserMemory(base, payloadSize));
				}
			}
			catch (...)
			{
				cleanup();
				throw;
			}
		}

		virtual ~SubGrabberManager()
		{
			cleanup();
		}

		void cleanup()
		{
			for (size_t i = 0; i < grabbers.length(); ++i)
				grabbers[i]->reallocBuffers(0);

			for (size_t i = 0; i < buffers.size(); ++i)
				free(buffers[i]);

			buffers.clear();
		}

		void go(size_t bufferCount)
		{
			size_t grabberCount = grabbers.length(); // set for a single octo board 

			bool triggerMode = !grabbers[0]->getInteger<RemoteModule>(query::available("SyncImg"));
			if (triggerMode)
			{
				grabbers[0]->setString<RemoteModule>("TriggerMode", "TriggerModeOn");   // camera in triggered mode
				grabbers[0]->setString<RemoteModule>("TriggerSource", "SWTRIGGER");     // source of trigger CXP
				grabbers[0]->setString<DeviceModule>("CameraControlMethod", "RC");      // tell grabber 0 to send trigger
				grabbers[0]->setString<DeviceModule>("ExposureReadoutOverlap", "True"); // camera needs 2 trigger to start
				grabbers[0]->setString<DeviceModule>("ErrorSelector", "All");
				grabbers[0]->setString<RemoteModule>("TimeStamp", "TSOff");
				//grabbers[0]->setString<RemoteModule>("AcquistionFrameRate", "4000");
				for (size_t i = 0; i < 1; i++)
				{
					grabbers[i]->setFloat<RemoteModule>("ExposureTime", 190);
					grabbers[i]->setString<DeviceModule>("CycleMinimumPeriod", "150.0");  // set the trigger rate to 250K Hz, 1/framerate
				}
				
			}
			else
			{
				// free-run mode with camera firmware older than end of September 2018
				// grabbers[0]->setString<RemoteModule>("SyncImg", "External");
				// NOT HANDLED
			}

			// start each grabber in reverse order
			for (size_t i = 0; i < grabberCount; ++i)
				grabbers[grabberCount - 1 - i]->start(bufferCount);

			const uint64_t timeout = 100000;
			for (size_t i = 0; i < bufferCount; ++i)
			{
				for (size_t ix = 0; ix < grabberCount; ++ix)
					grabbers[ix]->processEvent<NewBufferData>(timeout);
			}

			// convert & save merged buffers
			for (size_t i = 0; i < 1; ++i)
			{
				FormatConverter::Auto rgb(converter, FormatConverter::OutputFormat("RGB8"), buffers[i], pixelFormat, width, fullHeight);
				rgb.saveToDisk("merged.NNN.bmp", i);
			}
		}

	public:
		EGrabbers<SubGrabber> grabbers;
	private:
		std::vector<uint8_t *> buffers;

		size_t width;
		size_t height;
		size_t fullHeight;
		std::string pixelFormat;
		
		FormatConverter converter;
	};
}

int main() {
	EGenTL genTL;
	const size_t BUFFER_COUNT = 50;

	SubGrabberManager subGrabberManager(genTL, BUFFER_COUNT, 512, 512, "Mono8");
	subGrabberManager.go(BUFFER_COUNT);
	
	return 0;
}