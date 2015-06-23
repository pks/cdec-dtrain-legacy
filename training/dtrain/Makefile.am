bin_PROGRAMS = dtrain dtrain_net dtrain_net_interface

dtrain_SOURCES = dtrain.cc dtrain.h sample.h score.h update.h
dtrain_LDADD   = ../../decoder/libcdec.a ../../klm/search/libksearch.a ../../mteval/libmteval.a ../../utils/libutils.a ../../klm/lm/libklm.a ../../klm/util/libklm_util.a ../../klm/util/double-conversion/libklm_util_double.a

dtrain_net_SOURCES = dtrain_net.cc dtrain_net.h dtrain.h sample.h score.h update.h
dtrain_net_LDADD   = ../../decoder/libcdec.a ../../klm/search/libksearch.a ../../mteval/libmteval.a ../../utils/libutils.a ../../klm/lm/libklm.a ../../klm/util/libklm_util.a ../../klm/util/double-conversion/libklm_util_double.a /fast_scratch/simianer/lfpe/nanomsg-0.5-beta/lib/libnanomsg.so

dtrain_net_interface_SOURCES = dtrain_net_interface.cc dtrain_net_interface.h dtrain.h sample_net_interface.h score_net_interface.h update.h
dtrain_net_interface_LDADD   = ../../decoder/libcdec.a ../../klm/search/libksearch.a ../../mteval/libmteval.a ../../utils/libutils.a ../../klm/lm/libklm.a ../../klm/util/libklm_util.a ../../klm/util/double-conversion/libklm_util_double.a /fast_scratch/simianer/lfpe/nanomsg-0.5-beta/lib/libnanomsg.so

AM_CPPFLAGS = -W -Wall -Wno-sign-compare -I$(top_srcdir)/utils -I$(top_srcdir)/decoder -I$(top_srcdir)/mteval -I/fast_scratch/simianer/lfpe/nanomsg-0.5-beta/include

