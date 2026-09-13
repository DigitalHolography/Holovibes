add_executable(holovibes_dependency_smoke dependencies.cc)
target_link_libraries(holovibes_dependency_smoke PRIVATE
    holovibes_headers Boost::program_options HDF5::HDF5
    opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio
    Qt6::Widgets Qt6::Charts Qt6::OpenGLWidgets Qt6::Network OpenGL::GL
)
add_custom_command(TARGET holovibes_dependency_smoke POST_BUILD
    COMMAND Qt6::windeployqt $<IF:$<CONFIG:Debug>,--debug,--release>
        --include-plugins qoffscreen "$<TARGET_FILE:holovibes_dependency_smoke>"
    VERBATIM
)
add_test(NAME dependencies COMMAND holovibes_dependency_smoke)
set_tests_properties(dependencies PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
