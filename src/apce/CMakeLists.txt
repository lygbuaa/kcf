cmake_minimum_required(VERSION 2.8)

set(APCE_LIB_SRC apce_criterion.cc apce_criterion.h)

add_library(apce STATIC ${APCE_LIB_SRC})
target_link_libraries(apce ${OpenCV_LIBS})
set_target_properties(apce PROPERTIES VERSION 1.0.0 SOVERSION 1)