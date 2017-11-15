#pragma once

#ifndef _HAS_AUTO_PTR_ETC
#define _HAS_AUTO_PTR_ETC 1
#endif // !_HAS_AUTO_PTR_ETC


#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>


#include <stdexcept>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
# include <boost/algorithm/string.hpp>
# include <boost/filesystem.hpp>
# include <boost/property_tree/ptree.hpp>
# include <boost/property_tree/ini_parser.hpp>
# include <cstring>
# include <QDesktopServices>
# include <QFileDialog>
# include <QMainWindow>
# include <QMessageBox>
# include <QShortcut>
# include <sys/stat.h>
# include <thread>
# include <sstream>
# include <boost/tokenizer.hpp>
# include <boost/program_options/options_description.hpp>
# include <boost/program_options/cmdline.hpp>
# include <boost/program_options/eof_iterator.hpp>
# include <boost/program_options/errors.hpp>
# include <boost/program_options/option.hpp>
# include <boost/program_options/parsers.hpp>
# include <boost/program_options/variables_map.hpp>
# include <boost/program_options/positional_options.hpp>
# include <boost/program_options/environment_iterator.hpp>
# include <boost/program_options/config.hpp>
# include <boost/program_options/value_semantic.hpp>
# include <boost/program_options/version.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>


#include <PCO_err.h>
#include <sc2_defs.h>

#include <string>
#include <fstream>
#include <algorithm>
#include <thread>
#include <boost/lexical_cast.hpp>

#include <glm\gtc\type_ptr.hpp>

#include <Windows.h>

#include <device_launch_parameters.h>

// C include
#include <cassert>

// Standard Library
#include <atomic>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <array>
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <functional>
#include <exception>

// Boost
#include <boost/filesystem.hpp>

// Qt
#include <QObject>
#include <QApplication>
#include <QEvent.h>
#include <QDesktopWidget.h>
#include <QProgressBar>
#include <QTextBrowser>
#include <QThread>
	// QOpenGL
	#include <QOpenGLWindow>
	#include <QOpenGLFunctions>
	#include <QOpenGLVertexArrayObject>
	#include <QOpenGLShaderProgram>

// CUDA
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cufft.h>

// GLM
#include <glm\gtc\matrix_transform.hpp>
