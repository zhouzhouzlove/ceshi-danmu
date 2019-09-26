file(GLOB_RECURSE face_pose_src "${CMAKE_CURRENT_SOURCE_DIR}/../modules/*.cpp")
file(GLOB_RECURSE face_pose_header "${CMAKE_CURRENT_SOURCE_DIR}/../modules/*.h")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../modules")

add_library(face_pose  SHARED  face_pose.cpp ${face_pose_src} ${face_pose_header} ${inferxlite_models_src})
if(BUILD_WITH_OPENBLAS)
	target_link_libraries(face_pose ${OpenCV_LIBS} inferxlite openblas z )
endif()
if(BUILD_WITH_PERFDNN)
	target_link_libraries(face_pose ${OpenCV_LIBS} inferxlite perfblas perfdnn  z )
endif()
if(ANDROID)
	target_link_libraries(face_pose log )
endif()
install(TARGETS face_pose  DESTINATION lib)

