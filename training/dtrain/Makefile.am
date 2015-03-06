bin_PROGRAMS = dtrain dtrain_net

dtrain_SOURCES = dtrain.cc dtrain.h sample.h score.h update.h
dtrain_LDADD   = ../../decoder/libcdec.a ../../klm/search/libksearch.a ../../mteval/libmteval.a ../../utils/libutils.a ../../klm/lm/libklm.a ../../klm/util/libklm_util.a ../../klm/util/double-conversion/libklm_util_double.a

dtrain_net_SOURCES = dtrain_net.cc dtrain_net.h dtrain.h sample.h score.h update.h
dtrain_net_LDADD   = ../../decoder/libcdec.a ../../klm/search/libksearch.a ../../mteval/libmteval.a ../../utils/libutils.a ../../klm/lm/libklm.a ../../klm/util/libklm_util.a ../../klm/util/double-conversion/libklm_util_double.a /usr/lib64/libnanomsg.so

AM_CPPFLAGS = -W -Wall -Wno-sign-compare -I$(top_srcdir)/utils -I$(top_srcdir)/decoder -I$(top_srcdir)/mteval

