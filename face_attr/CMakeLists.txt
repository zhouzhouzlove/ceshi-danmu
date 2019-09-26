file(GLOB_RECURSE gender_judge_src "${CMAKE_CURRENT_SOURCE_DIR}/../utils/*.cpp")
file(GLOB_RECURSE gender_judge_header "${CMAKE_CURRENT_SOURCE_DIR}/../utils/*.h")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../utils")

add_library(face_attr  SHARED  face_attr.cpp ${gender_judge_src} ${gender_judge_header})
if(BUILD_WITH_OPENBLAS)
	target_link_libraries(face_attr ${OpenCV_LIBS} inferxlite openblas z)
endif()
if(BUILD_WITH_PERFDNN)
	target_link_libraries(face_attr ${OpenCV_LIBS} inferxlite perfblas perfdnn  z)
endif()
if(ANDROID)
	 target_link_libraries(face_attr log)
endif()
install(TARGETS face_attr  DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../")

