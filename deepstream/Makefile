################################################################################
# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
CXX:= g++

SRCS:= nvocdrlib_impl.cpp

INCS:= $(wildcard *.h)
LIB:=libnvocdr_impl.so

NVDS_VERSION:=6.2

CFLAGS+= -fPIC -DDS_VERSION=\"6.2.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I /opt/nvidia/deepstream/deepstream/sources/includes -I /opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdsvideotemplate/includes \
	 -I ../include 
	 
# OPENCV_INC = -I /usr/include/opencv4/
# OPENCV_LIB = -lopencv_core -lopencv_imgcodecs
# CFLAGS += $(OPENCV_INC)

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

ifeq ($(TARGET_DEVICE),aarch64)
	LIBS := -L/usr/lib/aarch64-linux-gnu/tegra
endif


LIBS += -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lcuda -ldl -lpthread \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
	-lnvbufsurface -lnvbufsurftransform -lnvdsbufferpool -L../ -lnvocdr \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

# LIBS += $(OPENCV_LIB)

OBJS:= $(SRCS:.cpp=.o)
PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

$(DEP): $(DEP_FILES)
	$(MAKE) -C customlib_impl/

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
