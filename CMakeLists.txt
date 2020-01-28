cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)

project(meta_vision_2020)



#FIXME: not work yet






# find the dependent package

    find_package(OpenCV REQUIRED)


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

set(META_VISION_SRCS
        Armor/ArmorDetector.cpp
        Coordata/package.cpp
        Main/main.cpp
        Driver/RMVideoCapture.cpp
        Pose/AngleSolver.cpp
        Serials/Serial.cpp)



add_executable(meta_sentry.exe 
                    Main/test_sentry.cpp ${META_VISION_SRCS})

target_link_libraries(meta_sentry.exe  ${OpenCV_LIBS})


add_executable(meta_infantry.exe 
                    Main/test_infantry.cpp ${META_VISION_SRCS})

target_link_libraries(meta_infantry.exe  ${OpenCV_LIBS})
