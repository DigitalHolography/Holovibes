#pragma once
// First, sort all the line
// To remove duplicated line, replace  ^(.*)(\r?\n\1)+$  by $1

// Because Boost use std::unary_function which has been removed in C++17
#ifndef _HAS_AUTO_PTR_ETC
#define _HAS_AUTO_PTR_ETC 1
#endif // !_HAS_AUTO_PTR_ETC

// Standard Library

#include <string>
#include <chrono>
#include <algorithm>
#include <array>
#include <atomic>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

// Qt
#include <QApplication>
#include <QDesktopServices>
#include <QDesktopWidget.h>
#include <QEvent.h>
#include <QFileDialog>
#include <QGroupBox>
#include <QMainWindow>
#include <QMessageBox>
#include <QObject>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWindow>
#include <QProgressBar>
#include <QShortcut>
#include <QTextBrowser>
#include <QThread>
#include <QVector>
#include <QtWidgets>

#include <qglobal.h>
#include <qmap.h>
#include <qrect.h>

// Qwt
#include <qwt_plot.h>
#include <qwt_plot_curve.h>

// Windows Kit
#include <Windows.h>
#include <direct.h>

#include <math.h>
#include <float.h>

// Boost
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
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

// C include
#include <stdint.h>
#include <sys/stat.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>

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
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
