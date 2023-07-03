
# CUDAVERSION=cuda-11.8
CUDAVERSION=cuda
CC = g++
CUCC = /usr/local/$(CUDAVERSION)/bin/nvcc -m64 -ccbin $(CC)
LIBDIR=lib64

TENSORRT_LIB_PATH = /usr/lib/x86_64-linux-gnu/
TENSORRT_INC_PATH = /usr/include/x86_64-linux-gnu/

OPENCV_LIB_PATH = /usr/lib/x86_64-linux-gnu/
OPENCV_INC_PATH = /usr/include/opencv4/
VERBOSE = 1
ifdef VERBOSE
AT=
else
AT=@
endif

ECHO = @echo 

SHELL=/bin/sh

OUT_PATH=.
OUTDIR=$(OUT_PATH)

define concat
$1$2$3$4$5$6$7$8
endef

#$(call make-depend,source-file,object-file,depend-file)
define make-depend
  $(AT)$(CC) -MM -MF $3 -MP -MT $2 $(COMMON_FLAGS) $1
endef

#########################

INCPATHS     =-I../include

LIBPATHS     =-L$(TENSORRT_LIB_PATH)
INCPATHS    +=-I$(TENSORRT_INC_PATH) -I../src

INCPATHS    +=-I/usr/local/$(CUDAVERSION)/include
LIBPATHS    +=-L/usr/local/$(CUDAVERSION)/$(LIBDIR)

INCPATHS 	+=-I$(OPENCV_INC_PATH)
LIBPATHS	+=-L$(OPENCV_LIB_PATH)
.SUFFIXES:

# COMMON_FLAGS += -Wall

COMMON_FLAGS += -std=c++11 $(INCPATHS)
COMMON_LD_FLAGS=$(LIBPATHS) -L$(OUTDIR)

OBJDIR    =$(call concat,$(OUTDIR),/chobj)
# TEST_OBJDIR =$(call concat,$(OUTDIR),/testobj)

COMMON_LIBS = -lnvinfer -lcudart -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lcublas -lnvinfer_plugin
COMMON_LD_FLAGS += $(COMMON_LIBS)

OUTNAME_RELEASE = nvocdr.cpython-38-x86_64-linux-gnu.so

SRC_ROOT=../src
CSRCS:= $(wildcard $(SRC_ROOT)/*.cpp)
COBJS:= $(notdir $(CSRCS:.cpp=.o))
OBJS=$(foreach obj, $(COBJS), $(OBJDIR)/$(obj))

CUSRCS:= $(wildcard $(SRC_ROOT)/*.cu)
TCUOBJS:= $(notdir $(CUSRCS:.cu=.o))
CUOBJS=$(foreach obj, $(TCUOBJS), $(OBJDIR)/$(obj))

# TEST_OBJS:= $(TEST_OBJDIR)/test_main.o

CFLAGS=$(COMMON_FLAGS)  
LFLAGS=$(COMMON_LD_FLAGS)

all: release
release : $(OUTDIR)/$(OUTNAME_RELEASE)
# release:
# 	$(ECHO) $(OBJS)
# 	$(ECHO) $(CUOBJS)
# 	$(ECHO) $(CFLAGS)
# 	$(ECHO) $(LFLAGS)

# $(OUTDIR)/$(OUTNAME_RELEASE) : $(OBJS) $(TEST_OBJS) $(CUOBJS)
$(OUTDIR)/$(OUTNAME_RELEASE) : $(OBJS) $(CUOBJS)
	$(ECHO) Linking: $@
	$(AT)$(CC) -O3 -Wall -shared $(CFLAGS) -fPIC  -I/usr/include/python3.8 -I/usr/local/lib/python3.8/dist-packages/pybind11/include pybind.cpp -o $@ $^ $(LFLAGS)  $(LIBS)


$(OBJDIR)/%.o: $(SRC_ROOT)/%.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -fPIC -c -o $@ $<

# $(TEST_OBJDIR)/%.o: %.cpp
# 	$(AT)if [ ! -d $(TEST_OBJDIR) ]; then mkdir -p $(TEST_OBJDIR); fi
# 	$(call make-depend,$<,$@,$(subst .o,.d,$@))
# 	$(ECHO) Compiling: $<
# 	$(AT)$(CC) $(CFLAGS) -fPIC -c -o $@ $<

$(OBJDIR)/%.o: $(SRC_ROOT)/%.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CUCC) --compiler-options '-fPIC' $(CFLAGS) -c -o $@ $<

clean:
	$(ECHO) Cleaning...
	$(AT)-rm -rf $(OBJDIR) $(TEST_OBJDIR) $(OUTDIR)/$(OUTNAME_RELEASE)

ifneq "$(MAKECMDGOALS)" "clean"
  -include $(OBJDIR)/*.d
endif

