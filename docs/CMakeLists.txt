find_package(Doxygen REQUIRED)


set(doxyfile ${CMAKE_CURRENT_SOURCE_DIR}/Doxygen/Doxyfile)
set(DOXYGEN_EXECUTABLE "C:/Program Files/doxygen/bin/doxygen.exe")

message("Doxygen build started.")

add_custom_target(doc 
                COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Generating documentation with Doxygen"
                VERBATIM)
