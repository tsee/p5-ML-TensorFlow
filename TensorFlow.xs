#define PERL_NO_GET_CONTEXT     /* we want efficiency */
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

#include "ppport.h"
#include "fix_inline.h"
#include <stdlib.h>
#include <sys/types.h>

/* Copy/pasting part of the c_api.h API here is obviously dumb,
 * but given how tensorflow is built (Python module) and installed,
 * it seems like expecting to find c_api.h anywhere as part of an
 * install is... */
/* On second thought, reproducing the API using the FFI isn't any more
 * maintainable/more compatible unless one parses the header and
 * generates code from there. So meh. */

/* FIXME: Sync with c_api.h */
typedef struct {
  const void* data;
  size_t length;
  void (*data_deallocator)(void* data, size_t length);
} TF_Buffer;

extern TF_Buffer* TF_NewBufferFromString(const void* proto, size_t proto_len);

/* Useful for passing *out* a protobuf. */
extern TF_Buffer* TF_NewBuffer();

extern void TF_DeleteBuffer(TF_Buffer*);

/* END silly copy of partial c_api.h */


void safefree_deallocator(void *data, size_t length)
{
  Safefree(data);
}

/* end c_api.h portion */

MODULE = ML::TensorFlow		PACKAGE = ML::TensorFlow::CAPI
PROTOTYPES: DISABLE

SV *
_make_perl_string_copy_from_opaque_string(SV *s_ptr, size_t len)
  CODE:
    RETVAL = newSVpvn((char *)SvIV(s_ptr), len);
  OUTPUT: RETVAL

MODULE = ML::TensorFlow		PACKAGE = ML::TensorFlow::Buffer
PROTOTYPES: DISABLE

UV
_get_struct_size()
  CODE:
    RETVAL = (UV)sizeof(TF_Buffer);
  OUTPUT: RETVAL

TF_Buffer*
new(CLASS, ...)
    char *CLASS
  CODE:
    if (items == 1) {
      RETVAL = TF_NewBuffer();
    }
    else if (items == 2) {
      STRLEN len;
      const char *str = SvPVbyte(ST(1), len);
      RETVAL = TF_NewBufferFromString((const void *)str, (size_t)len);
    }
    else {
      croak("Invalid number of arguments to buffer constructor");
    }
  OUTPUT: RETVAL

void
DESTROY(self)
    TF_Buffer* self
  CODE:
    TF_DeleteBuffer(self);

SV*
get_data_copy(self)
    TF_Buffer* self
  PREINIT:
  CODE:
    RETVAL = newSVpvn((const char *)self->data, (STRLEN)self->length);
  OUTPUT: RETVAL

SV*
get_data_view(self)
    TF_Buffer* self
  PREINIT:
  CODE:
    RETVAL = newSV(0);
    /* read-only view - user beware of the memory management consequences */
    SvUPGRADE(RETVAL, SVt_PV);
    SvPOK_on(RETVAL);
    SvPV_set(RETVAL, (char *)self->data);
    SvCUR_set(RETVAL, (STRLEN)self->length);
    SvLEN_set(RETVAL, 0);
    SvREADONLY_on(RETVAL);
  OUTPUT: RETVAL

void
set_data(self, data)
    TF_Buffer *self
    SV *data
  PREINIT:
    STRLEN len;
    const char *str;
  CODE:
    self->data_deallocator((void *)self->data, self->length);
    self->data_deallocator = &safefree_deallocator;

    str = SvPVbyte(data, len);
    self->length = (size_t)len;

    Newx(self->data, len, char);
    Copy(str, self->data, len, char);


