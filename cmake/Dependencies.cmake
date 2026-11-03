find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(spdlog CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS algorithm lexical_cast program_options property_tree tokenizer)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Qt6 CONFIG REQUIRED COMPONENTS Widgets Core Charts OpenGL OpenGLWidgets Network)

# These headers are used throughout the backend, UI and camera interfaces.
add_library(holovibes_headers INTERFACE)
target_link_libraries(holovibes_headers INTERFACE
    nlohmann_json::nlohmann_json
    glm::glm
    Boost::algorithm
    Boost::lexical_cast
    Boost::property_tree
    Boost::tokenizer
    spdlog::spdlog_header_only
)
