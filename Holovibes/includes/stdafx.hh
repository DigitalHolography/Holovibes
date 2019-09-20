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
* Precompiled header. Put here all the external includes that aren't used in a cuda file
  to avoid recompiling it each time. */

#pragma once
// First, sort all the line
// To remove duplicated line, replace  ^(.*)(\r?\n\1)+$  by $1

// Because Boost uses std::unary_function which has been removed in C++17
#ifndef _HAS_AUTO_PTR_ETC
#define _HAS_AUTO_PTR_ETC 1
#endif // !_HAS_AUTO_PTR_ETC

// Because Boost uses std::iterator which is deprecated in C++17
#ifndef _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#endif

// Because Boost uses std::allocator which is deprecated in C++17
#ifndef _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING
#define _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING
#endif

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

// Standard Library

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

// C include
#include <stdint.h>
#include <sys/stat.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>

// Qt
#include <QAction>
#include <QApplication>
#include <QDesktopServices>
#include <QDesktopWidget.h>
#include <QDragEnterEvent>
#include <QEvent.h>
#include <QFileDialog>
#include <QFont>
#include <QGroupBox>
#include <QGuiApplication>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QMouseEvent>
#include <QObject>
#include <QOpenGLFunctions>
#include <QOpenGLPaintDevice>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWindow>
#include <QPainter>
#include <QPen>
#include <QProgressBar>
#include <QRect>
#include <QScreen>
#include <QShortcut>
#include <QTextBrowser>
#include <QTextDocument>
#include <QThread>
#include <QVector>
#include <QWheelEvent>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtWidgets>

#include <qabstracttextdocumentlayout.h>
#include <qglobal.h>
#include <qmap.h>
#include <qrect.h>
#include <qtextdocument.h>

// Windows Kit
#include <Windows.h>
#include <direct.h>

// Boost
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/config.hpp>
#include <boost/program_options/environment_iterator.hpp>
#include <boost/program_options/eof_iterator.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/version.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/tokenizer.hpp>

// CUDA
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>