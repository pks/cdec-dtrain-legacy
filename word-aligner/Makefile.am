bin_PROGRAMS = fast_align binderiv net_fa

fast_align_SOURCES = fast_align.cc ttables.cc da.h ttables.h
fast_align_LDADD = ../utils/libutils.a
fast_align_LDFLAGS = $(STATIC_FLAGS)

binderiv_SOURCES = binderiv.cc
binderiv_LDADD = ../utils/libutils.a

net_fa_SOURCES = net_fa.cc ttables.cc da.h ttables.h nn.hpp
net_fa_LDADD = ../utils/libutils.a
net_fa_LDFLAGS = $(STATIC_FLAGS) /usr/lib64/libnanomsg.so

EXTRA_DIST = aligner.pl ortho-norm support makefiles stemmers

AM_CPPFLAGS = -W -Wall -I$(top_srcdir) -I$(top_srcdir)/utils -I$(top_srcdir)/training
