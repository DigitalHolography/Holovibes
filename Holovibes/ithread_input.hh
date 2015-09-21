
#ifndef ITHREAD_INPUT_HH
#define ITHREAD_INPUT_HH

namespace holovibes
{
	class IThreadInput
	{
	protected:
		IThreadInput();
	public:
		virtual ~IThreadInput();
		bool stop_requested_;
	};
}
#endif