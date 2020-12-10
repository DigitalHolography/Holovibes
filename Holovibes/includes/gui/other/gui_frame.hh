/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file
*
* Contains the overloading of QFrame. */
#pragma once

#include <QFrame>
#include <QObject>

namespace holovibes
{
	namespace gui
	{
		/*! \brief QFrame overload, used to hide and show parts of the GUI. */
		class Frame : public QFrame
		{
			Q_OBJECT

		public:
			/*! \brief Frame constructor
			** \param parent Qt parent
			*/
			Frame(QWidget* parent = nullptr);
			/*! \brief Frame destructor */
			~Frame();

			public slots:
			/*! \brief Show or hide Frame */
			void ShowOrHide();
		};
	}
}