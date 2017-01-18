#pragma once

# include <QOpenGLWidget>
# include <QOpenGLFunctions.h>

namespace gui {

	// To Do
	class HoloWidget : public QOpenGLWidget, protected QOpenGLFunctions
	{
		public:
			virtual ~HoloWidget() {}

		protected:
			const unsigned int	DisplayFramerate = 30;


	};

}