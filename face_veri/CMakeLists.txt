file(GLOB_RECURSE face_veri_src "${CMAKE_CURRENT_SOURCE_DIR}/../modules/*.cpp")
file(GLOB_RECURSE face_veri_header "${CMAKE_CURRENT_SOURCE_DIR}/../modules/*.h")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../modules")

add_library(face_verify  SHARED  face_veri.cpp ${face_veri_src} ${face_veri_header} ${inferxlite_models_src})
if(BUILD_WITH_OPENBLAS)
	target_link_libraries(face_verify ${OpenCV_LIBS} inferxlite openblas z )
endif()
if(BUILD_WITH_PERFDNN)
	target_link_libraries(face_verify ${OpenCV_LIBS} inferxlite perfblas perfdnn  z )
endif()
if(ANDROID)
	target_link_libraries(face_verify log )
endif()
install(TARGETS face_verify  DESTINATION lib)

