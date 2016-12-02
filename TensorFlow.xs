#define PERL_NO_GET_CONTEXT     /* we want efficiency */
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

#include "ppport.h"
#include "fix_inline.h"
#include <stdlib.h>
#include <sys/types.h>


MODULE = ML::TensorFlow		PACKAGE = ML::TensorFlow
PROTOTYPES: DISABLE


