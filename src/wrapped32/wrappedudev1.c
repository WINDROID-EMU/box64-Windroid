#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#include <dlfcn.h>
#include <stdarg.h>

#include "wrappedlibs.h"

#include "wrapper32.h"
#include "bridge.h"
#include "librarian/library_private.h"
#include "x64emu.h"
#include "debug.h"
#include "myalign32.h"
#include "callback.h"
#include "emu/x64emu_private.h"

static const char* udev1Name = "libudev.so.1";
#define LIBNAME udev1
// fallback to 0 version... Not sure if really correct (probably not)
#define ALTNAME "libudev.so.0"

#define ADDED_FUNCTIONS()           \

#include "generated/wrappedudev1types32.h"

#include "wrappercallback32.h"

#define SUPER() \
GO(0)   \
GO(1)   \
GO(2)   \
GO(3)   \
GO(4)

// log_fn ...
#define GO(A)   \
static uintptr_t my_log_fn_fct_##A = 0;                                                                 \
static void my_log_fn_##A(void* udev, int p, void *f, int l, void* fn, void* fmt, va_list args)         \
{                                                                                                       \
    static char buff[1024];                                                                             \
    vsnprintf(buff, 1023, fmt, args);                                                                   \
    buff[1023]=0;                                                                                       \
    RunFunction(my_log_fn_fct_##A, 7, udev, p, f, l, fn, "%s", buff);                                   \
}
SUPER()
#undef GO
static void* find_log_fn_Fct(void* fct)
{
    if(!fct) return fct;
    if(GetNativeFnc((uintptr_t)fct))  return GetNativeFnc((uintptr_t)fct);
    #define GO(A) if(my_log_fn_fct_##A == (uintptr_t)fct) return my_log_fn_##A;
    SUPER()
    #undef GO
    #define GO(A) if(my_log_fn_fct_##A == 0) {my_log_fn_fct_##A = (uintptr_t)fct; return my_log_fn_##A; }
    SUPER()
    #undef GO
    printf_log(LOG_NONE, "Warning, no more slot for udev1 log_fn callback\n");
    return NULL;
}
#undef SUPER

EXPORT void my32_udev_set_log_fn(x64emu_t* emu, void* udev, void* f)
{
    my->udev_set_log_fn(udev, find_log_fn_Fct(f));
}

#include "wrappedlib_init32.h"