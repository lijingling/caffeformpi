# Install script for directory: /ssd/lijingling/resnet_MSRA/caffe-mpi/python

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/ssd/lijingling/resnet_MSRA/caffe-mpi/build/install")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE FILE FILES
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/classify.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/draw_net.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/detect.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/gen_bn_inference.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/convert_to_fully_conv.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/polyak_average.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/bn_convert_style.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/requirements.txt"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE FILE FILES
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/caffe/detector.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/caffe/net_spec.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/caffe/pycaffe.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/caffe/__init__.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/caffe/draw.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/caffe/classifier.py"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/caffe/io.py"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    FILE(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so"
         RPATH "/ssd/lijingling/resnet_MSRA/caffe-mpi/build/install/lib:/usr/local/lib:/usr/local/cuda/lib64:/usr/local/nccl/lib")
  ENDIF()
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE SHARED_LIBRARY FILES "/ssd/lijingling/resnet_MSRA/caffe-mpi/build/lib/_caffe.so")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    FILE(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so"
         OLD_RPATH "/ssd/lijingling/resnet_MSRA/caffe-mpi/build/lib:/usr/local/lib:/usr/local/cuda/lib64:/usr/local/nccl/lib::::::::"
         NEW_RPATH "/ssd/lijingling/resnet_MSRA/caffe-mpi/build/install/lib:/usr/local/lib:/usr/local/cuda/lib64:/usr/local/nccl/lib")
    IF(CMAKE_INSTALL_DO_STRIP)
      EXECUTE_PROCESS(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    ENDIF(CMAKE_INSTALL_DO_STRIP)
  ENDIF()
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE DIRECTORY FILES
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/caffe/imagenet"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/caffe/proto"
    "/ssd/lijingling/resnet_MSRA/caffe-mpi/python/caffe/test"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

