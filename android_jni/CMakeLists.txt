file(GLOB_RECURSE jni_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE  jni_header "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_library(native-lib  SHARED  ${jni_src} ${jni_header} ${face_detector_file} ${face_veri_file} ${face_attr_file} ${face_detect_feature_src} ${face_detect_feature_header})
if(BUILD_WITH_OPENBLAS)
	target_link_libraries(native-lib ${OpenCV_LIBS} inferxlite openblas z )
endif()
if(BUILD_WITH_PERFDNN)
	target_link_libraries(native-lib ${OpenCV_LIBS} inferxlite perfblas perfdnn  z )
endif()
if(ANDROID)
	target_link_libraries(native-lib log jnigraphics android)
endif()
install(TARGETS native-lib  DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../")

