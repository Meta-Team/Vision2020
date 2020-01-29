cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

project(meta_vision_2020)



#FIXME: not work yet






# find the dependent package

    find_package(OpenCV REQUIRED)
    
    set(INC_DIR /usr/local/include)
    set(LINK_DIR /usr/local/lib)
    include_directories(${INC_DIR})
    link_directories(${LINK_DIR})

include_directories(
    Armor
    CoorData
    Driver
    General
    Main
    Pose
    Rune
    Serials
)

aux_source_directory(Armor      ARMOR_SRCS)
aux_source_directory(Driver     DRIVER_SRCS)
aux_source_directory(General    GENERAL_SRCS)
aux_source_directory(Pose       POSE_SRCS)
aux_source_directory(Rune       RUNE_SRCS)
aux_source_directory(Serials    SERIALS_SRCS)


set(META_VISION_SRCS
        Main/main.cpp
        Main/ImgProdCons.cpp
        ${ARMOR_SRCS}
        ${DRIVER_SRCS}
        ${GENERAL_SRCS}
        ${POSE_SRCS}
        ${RUNE_SRCS}
        ${SERIALS_SRCS})



add_executable(meta_sentry.exe 
                    Main/test_sentry.cpp ${META_VISION_SRCS})

target_link_libraries(meta_sentry.exe  ${OpenCV_LIBS} v4l2 pthread)


add_executable(meta_infantry.exe 
                    Main/test_infantry.cpp ${META_VISION_SRCS})

target_link_libraries(meta_infantry.exe  ${OpenCV_LIBS}  v4l2 pthread)
