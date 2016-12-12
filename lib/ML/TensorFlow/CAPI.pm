package ML::TensorFlow::CAPI;
use 5.14.2;
use warnings;

our $VERSION = '1.01';

require ML::TensorFlow;
use FFI::Platypus;
use FFI::CheckLib ();
use FFI::Platypus::Record;

use Exporter 'import';
our @EXPORT_OK;
our %EXPORT_TAGS = (all => \@EXPORT_OK);

my %TF_Code_Enum;
my %TF_DataType_Enum;

BEGIN: {
  %TF_DataType_Enum = (
    TF_FLOAT => 1,
    TF_DOUBLE => 2,
    TF_INT32 => 3,  # Int32 tensors are always in 'host' memory.
    TF_UINT8 => 4,
    TF_INT16 => 5,
    TF_INT8 => 6,
    TF_STRING => 7,
    TF_COMPLEX64 => 8,  # Single-precision complex
    TF_COMPLEX => 8,    # Old identifier kept for API backwards compatibility
    TF_INT64 => 9,
    TF_BOOL => 10,
    TF_QINT8 => 11,     # Quantized int8
    TF_QUINT8 => 12,    # Quantized uint8
    TF_QINT32 => 13,    # Quantized int32
    TF_BFLOAT16 => 14,  # Float32 truncated to 16 bits.  Only for cast ops.
    TF_QINT16 => 15,    # Quantized int16
    TF_QUINT16 => 16,   # Quantized uint16
    TF_UINT16 => 17,
    TF_COMPLEX128 => 18,  # Double-precision complex
    TF_HALF => 19,
    TF_RESOURCE => 20,
  );

  %TF_Code_Enum = (
    TF_OK => 0,
    TF_CANCELLED => 1,
    TF_UNKNOWN => 2,
    TF_INVALID_ARGUMENT => 3,
    TF_DEADLINE_EXCEEDED => 4,
    TF_NOT_FOUND => 5,
    TF_ALREADY_EXISTS => 6,
    TF_PERMISSION_DENIED => 7,
    TF_UNAUTHENTICATED => 16,
    TF_RESOURCE_EXHAUSTED => 8,
    TF_FAILED_PRECONDITION => 9,
    TF_ABORTED => 10,
    TF_OUT_OF_RANGE => 11,
    TF_UNIMPLEMENTED => 12,
    TF_INTERNAL => 13,
    TF_UNAVAILABLE => 14,
    TF_DATA_LOSS => 15,
  );

  push @EXPORT_OK, qw(TF_DataType_Enum TF_Code_Enum);
} # end BEGIN

use constant (TF_DataType_Enum => \%TF_DataType_Enum);
use constant (TF_Code_Enum => \%TF_Code_Enum);


# Global Init
our $FFI = FFI::Platypus->new;
$FFI->lib(FFI::CheckLib::find_lib_or_exit(lib => 'tensorflow'));



# just some named types for the FFI to make it more readable
my $TF_Status_Ptr                 = "opaque";
my $TF_Tensor_Ptr                 = "opaque";
my $TF_SessionOptions_Ptr         = "opaque";
my $TF_Session_Ptr                = "opaque";
my $TF_Buffer_Ptr                 = "opaque";
my $TF_Graph_Ptr                  = "opaque";
my $TF_Library_Ptr                = "opaque";
my $TF_ImportGraphDefOptions_Ptr  = "opaque";

my $TF_Code_Enum_t     = "int";
my $TF_DataType_Enum_t = "int";

use constant TF_BUFFER_SIZE => ML::TensorFlow::Buffer::_get_struct_size();
my $TF_Buffer_Opaque = "record(" . TF_BUFFER_SIZE . ")";

# Status API
$FFI->attach( "TF_NewStatus",           [] => $TF_Status_Ptr );
$FFI->attach( "TF_DeleteStatus",        [$TF_Status_Ptr] => "void" );
$FFI->attach( "TF_SetStatus",           [$TF_Status_Ptr, $TF_Code_Enum_t, "string"] => "void" );
$FFI->attach( "TF_GetCode",             [$TF_Status_Ptr] => $TF_Code_Enum_t );
$FFI->attach( "TF_Message",             [$TF_Status_Ptr] => "string" );


# Buffer API
# See TensorFLow.xs
#$FFI->attach( "TF_NewBufferFromString", ['string', 'size_t'] => $TF_Buffer_Ptr );
#$FFI->attach( "TF_NewBuffer",           [] => $TF_Buffer_Ptr );
#$FFI->attach( "TF_DeleteBuffer",        [$TF_Buffer_Ptr] => 'void' );

# WTF:?
#extern TF_Buffer TF_GetBuffer(TF_Buffer* buffer);
#typedef struct {
#  const void* data;
#  size_t length;
#  void (*data_deallocator)(void* data, size_t length);
#} TF_Buffer;


# Tensor API
#TF_DataType, const int64_t* dims, int num_dims,
#                             void* data, size_t len,
#                             void (*deallocator)(void* data, size_t len,
#                                                 void* arg),
#                             void* deallocator_arg);
#exports.TF_Destructor = ffi.Callback('void', ['void*', 'size_t', 'void*'], function(data, len, arg) {});

$FFI->type('(string,size_t,opaque)->void' => 'tensor_dealloc_closure_t');

$FFI->attach(
  'TF_NewTensor',
  [
    $TF_DataType_Enum_t,
    'sint64[]', 'int', # dim sizes, ndims
    'string', 'size_t', # data, data len in bytes
    'tensor_dealloc_closure_t', 'opaque', # deallocator callback, deallocator arg
  ],
  $TF_Tensor_Ptr
);

$FFI->attach( 'TF_AllocateTensor',      [$TF_DataType_Enum_t, 'sint64[]', 'int', 'size_t'] => $TF_Tensor_Ptr );
$FFI->attach( 'TF_DeleteTensor',        [$TF_Tensor_Ptr] => 'void' );
$FFI->attach( 'TF_TensorType',          [$TF_Tensor_Ptr] => $TF_DataType_Enum_t );
$FFI->attach( 'TF_NumDims',             [$TF_Tensor_Ptr] => 'int' );
$FFI->attach( 'TF_Dim',                 [$TF_Tensor_Ptr, 'int'] => 'sint64' );
$FFI->attach( 'TF_TensorByteSize',      [$TF_Tensor_Ptr] => 'size_t' );

# warning: no encapsulation whatsoever to this one...
$FFI->attach( 'TF_TensorData',          [$TF_Tensor_Ptr] => 'opaque');

# String-encode/decode stuff
# TODO
#extern size_t TF_StringEncode(const char* src, size_t src_len, char* dst,
#                              size_t dst_len, TF_Status* status);
#extern size_t TF_StringDecode(const char* src, size_t src_len, const char** dst,
#                              size_t* dst_len, TF_Status* status);
#extern size_t TF_StringEncodedSize(size_t len);


# SessionOptions API
$FFI->attach( 'TF_NewSessionOptions',   [] => $TF_SessionOptions_Ptr );
$FFI->attach( 'TF_SetTarget',           [$TF_SessionOptions_Ptr, 'string'] => 'void' );
#$FFI->attach( 'TF_SetConfig',           [$TF_SessionOptions_Ptr, 'void*', 'size_t', $TF_Status_Ptr] => 'void' );
$FFI->attach( 'TF_SetConfig',           [$TF_SessionOptions_Ptr, 'string', 'size_t', $TF_Status_Ptr] => 'void' );
$FFI->attach( 'TF_DeleteSessionOptions', [$TF_SessionOptions_Ptr] => 'void' );


# Graph API
$FFI->attach( 'TF_NewGraph',             [] => $TF_Graph_Ptr);
$FFI->attach( 'TF_DeleteGraph',          [$TF_Graph_Ptr] => 'void');

# ImportGraphDef(Options) API
$FFI->attach( 'TF_NewImportGraphDefOptions', [] => $TF_ImportGraphDefOptions_Ptr );
$FFI->attach( 'TF_DeleteImportGraphDefOptions', [$TF_ImportGraphDefOptions_Ptr] => 'void' );
$FFI->attach( 'TF_ImportGraphDefOptionsSetPrefix', [$TF_ImportGraphDefOptions_Ptr, 'string'] => 'void' );
$FFI->attach( 'TF_GraphImportGraphDef',
              [ $TF_Graph_Ptr,
                $TF_Buffer_Ptr,
                $TF_ImportGraphDefOptions_Ptr,
                $TF_Status_Ptr ]
              => 'void' );

# TODO the whole graph/operation shebang


# Session API
# FIXME this seems to leak deep inside tensorflow.
$FFI->attach( 'TF_NewSession',          [$TF_Graph_Ptr, $TF_SessionOptions_Ptr, $TF_Status_Ptr] => $TF_Session_Ptr );
$FFI->attach( 'TF_CloseSession',        [$TF_Session_Ptr, $TF_Status_Ptr] => 'void' );
$FFI->attach( 'TF_DeleteSession',       [$TF_Session_Ptr, $TF_Status_Ptr] => 'void' );


# Running things API
# TODO
# extern void TF_SessionRun(TF_Session* session,
#                           // RunOptions
#                           const TF_Buffer* run_options,
#                           // Input tensors
#                           const TF_Output* inputs,
#                           TF_Tensor* const* input_values, int ninputs,
#                           // Output tensors
#                           const TF_Output* outputs, TF_Tensor** output_values,
#                           int noutputs,
#                           // Target operations
#                           const TF_Operation* const* target_opers, int ntargets,
#                           // RunMetadata
#                           TF_Buffer* run_metadata,
#                           // Output status
#                           TF_Status*);

# There's also TF_SessionPRunSetup,TF_SessionPRun, which is marked as experimental

# TF_Library API
$FFI->attach( "TF_LoadLibrary",         ['string', $TF_Status_Ptr] => $TF_Library_Ptr );
# returning straight up buffer (not ptr)
$FFI->attach( "TF_GetOpList",           [$TF_Library_Ptr] => $TF_Buffer_Opaque );
$FFI->attach( "TF_DeleteLibraryHandle", [$TF_Library_Ptr] => 'void' );
$FFI->attach( "TF_GetAllOpList",        [] => $TF_Buffer_Ptr );


1;

__END__

=head1 NAME

ML::TensorFlow::CAPI - Internal CAPI wrapper

=head1 SYNOPSIS

  use ML::TensorFlow;

=head1 DESCRIPTION

=head2 EXPORT

=head1 FUNCTIONS

=head1 SEE ALSO

=head1 AUTHOR

Steffen Mueller, E<lt>smueller@cpan.orgE<gt>

=head1 COPYRIGHT AND LICENSE

Copyright (C) 2016 by Steffen Mueller

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.0 or,
at your option, any later version of Perl 5 you may have available.

=cut

