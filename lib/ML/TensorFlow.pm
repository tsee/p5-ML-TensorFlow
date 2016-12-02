package ML::TensorFlow;
use 5.14.2;
use warnings;

require Exporter;

our $VERSION = '0.01';

use Scalar::Util ();
require bytes;

#require XSLoader;
#XSLoader::load('ML::TensorFlow', $VERSION);

use Exporter 'import';
our @EXPORT_OK;
our %EXPORT_TAGS = (all => \@EXPORT_OK);

my %TF_Code;
my %TF_DataType;

BEGIN: {
  %TF_DataType = (
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

  %TF_Code = (
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

  push @EXPORT_OK, qw(TF_DataType TF_Code);
} # end BEGIN

use constant (TF_DataType => \%TF_DataType);
use constant (TF_Code => \%TF_DataType);
use constant {
  Status         => "ML::TensorFlow::Status",
  Session        => "ML::TensorFlow::Session",
  SessionOptions => "ML::TensorFlow::SessionOptions",
  Tensor         => "ML::Tensor::Tensor",
};
push @EXPORT_OK, qw(Status SessionOptions Session Tensor);


use FFI::Platypus;
use FFI::CheckLib (); # qw( find_lib_or_exit );

package ML::TensorFlow::CAPI {

  # just some named types for the FFI to make it more readable
  my $TF_Status         = "opaque";
  my $TF_Tensor         = "opaque";
  my $TF_SessionOptions = "opaque";
  my $TF_Session        = "opaque";

  my $TF_Code_t = "int";
  my $TF_DataType_t = "int";

  my $ffi = FFI::Platypus->new;
  $ffi->lib(FFI::CheckLib::find_lib_or_exit(lib => 'tensorflow'));

  # Status API
  $ffi->attach( "TF_NewStatus", [] => $TF_Status );
  $ffi->attach( "TF_DeleteStatus", [$TF_Status] => "void" );
  $ffi->attach( "TF_SetStatus", [$TF_Status, $TF_Code_t, "string"] => "void" );
  $ffi->attach( "TF_GetCode", [$TF_Status] => $TF_Code_t );
  $ffi->attach( "TF_Message", [$TF_Status] => "string" );

  # SessionOptions API
  $ffi->attach( 'TF_NewSessionOptions', [] => $TF_SessionOptions );
  $ffi->attach( 'TF_SetTarget', [$TF_SessionOptions, 'string'] => 'void' );
  #$ffi->attach( 'TF_SetConfig', [$TF_SessionOptions, 'void*', 'size_t', $TF_Status] => 'void' );
  $ffi->attach( 'TF_SetConfig', [$TF_SessionOptions, 'string', 'size_t', $TF_Status] => 'void' );
  $ffi->attach( 'TF_DeleteSessionOptions', [$TF_SessionOptions] => 'void' );

  # Session API
  $ffi->attach( 'TF_NewSession', [$TF_SessionOptions, $TF_Status] => $TF_Session );
  $ffi->attach( 'TF_CloseSession', [$TF_Session, $TF_Status] => 'void' );
  $ffi->attach( 'TF_DeleteSession', [$TF_Session, $TF_Status] => 'void' );
  #$ffi->attach( 'TF_ExtendGraph', [$TF_Session, 'void*', 'size_t', $TF_Status] => 'void' );
  #$ffi->attach( 'TF_ExtendGraph', [$TF_Session, 'string', 'size_t', $TF_Status] => 'void' );

  # Tensor API
  #TF_DataType, const int64_t* dims, int num_dims,
  #                             void* data, size_t len,
  #                             void (*deallocator)(void* data, size_t len,
  #                                                 void* arg),
  #                             void* deallocator_arg);
  #exports.TF_Destructor = ffi.Callback('void', ['void*', 'size_t', 'void*'], function(data, len, arg) {});

  $ffi->type('(opaque,size_t,opaque)->void' => 'tensor_dealloc_closure_t');
  $ffi->attach(
    'TF_NewTensor',
    [
      $TF_DataType_t,
      'sint64[]', 'int', # dim sizes, ndims
      'opaque', 'size_t', # data, data len in bytes
      'tensor_dealloc_closure_t', 'opaque', # deallocator callback, deallocator arg
    ],
    $TF_Tensor
  );

  $ffi->attach( 'TF_DeleteTensor', [$TF_Tensor] => 'void' );
  $ffi->attach( 'TF_TensorType', [$TF_Tensor] => $TF_DataType_t );
  $ffi->attach( 'TF_NumDims', [$TF_Tensor, 'int'] => 'int' );
  $ffi->attach( 'TF_Dim', [$TF_Tensor] => 'sint64' );
  $ffi->attach( 'TF_TensorByteSize', [$TF_Tensor] => 'size_t' );

  # warning: no encapsulation...
  $ffi->attach( 'TF_TensorData', [$TF_Tensor] => 'opaque');

};

package ML::TensorFlow::Status {
  sub new {
    my ($class) = @_;
    my $s = ML::TensorFlow::CAPI::TF_NewStatus();
    my $self = bless(\$s => $class);
    $self->set_status(0, "");
    return $self;
  }

  sub DESTROY {
    my ($self) = @_;
    ML::TensorFlow::CAPI::TF_DeleteStatus($$self);
  }

  sub is_ok { 
    my ($self) = @_;
    my $code = $self->get_code;
    return($code == TF_Code()->{TF_OK});
  }

  sub set_status {
    my ($self, $code, $message) = @_;
    ML::TensorFlow::CAPI::TF_SetStatus($$self, 0+$code, $message);
  }

  sub get_code {
    my ($self) = @_;
    return ML::TensorFlow::CAPI::TF_GetCode($$self);
  }

  sub get_message {
    my ($self) = @_;
    return ML::TensorFlow::CAPI::TF_Message($$self);
  }
}; # end Status

package ML::TensorFlow::SessionOptions {
  sub new {
    my ($class) = @_;
    my $s = ML::TensorFlow::CAPI::TF_NewSessionOptions();
    my $self = bless(\$s => $class);

    $self->set_target("");
    $self->set_config("");
    return $self;
  }

  sub DESTROY {
    my ($self) = @_;
    ML::TensorFlow::CAPI::TF_DeleteSessionOptions($$self);
  }

  sub set_target {
    my ($self, $target) = @_;
    ML::TensorFlow::CAPI::TF_SetTarget($$self, $target); # FIXME don't know FFI::Platypus well enough, but normally in XS calls, there's a risk because Perl strings aren't guaranteed to be NUL terminated!
  }

  # TODO test
  sub set_config {
    my ($self, $protoblob) = @_;
    my $status = ML::TensorFlow::Status->new();
    ML::TensorFlow::CAPI::TF_SetConfig($$self, $protoblob, bytes::length($protoblob), $$status);
    return $status;
  }

}; # end SessionOptions


package ML::TensorFlow::Session {
  sub new {
    my ($class, $graph, $sessopt) = @_;

    if (!Scalar::Util::blessed($graph) || !$sessopt->isa("ML::TensorFlow::Graph")) {
      Carp::croak("Need a Graph object");
    }

    if (!Scalar::Util::blessed($sessopt) || !$sessopt->isa("ML::TensorFlow::SessionOptions")) {
      Carp::croak("Need a SessionOptions object");
    }

    my $status = ML::TensorFlow::Status->new;
    my $s = ML::TensorFlow::CAPI::TF_NewSession($$graph, $$sessopt, $$status);
    if (not $status->is_ok) {
      die("Failed to create new TF session: " . $status->get_message);
    }

    my $self = bless(\$s => $class);
    return $self;
  }

  sub DESTROY {
    my ($self) = @_;
    my $status = ML::TensorFlow::Status->new; # Mocking up status
    ML::TensorFlow::CAPI::TF_DeleteSession($$self, $$status);
    return;
  }

  sub close {
    my ($self) = @_;
    my $status = ML::TensorFlow::Status->new;
    ML::TensorFlow::CAPI::TF_CloseSession($$self, $$status);
    return $status;
  }

  #$ffi->attach( 'TF_ExtendGraph', [$TF_Session, 'void*', 'size_t', $TF_Status] => 'void' );
  #sub extend_graph {
  #  my ($self, $binary_data) = @_;
  #  my $status = ML::TensorFlow::Status->new;
  #  ML::TensorFlow::CAPI::TF_ExtendGraph($$self, $binary_data, bytes::length($binary_data), $$status);
  #  return $status;
  #}
}; # end Session


package ML::TensorFlow::Tensor {

  my $dealloc_closure = sub {}; # memory managed by Perl (?)
  sub new {
    my ($class, $dims, $datablob) = @_;

    my $s = ML::TensorFlow::CAPI::TF_NewTensor(
      $dims, scalar(@$dims),
      $datablob, bytes::length($datablob),
      $dealloc_closure, 0 # as in: NULL 
    );

    my $self = bless(\$s => $class);
    return $self;
  }

  sub DESTROY {
    my ($self) = @_;
    ML::TensorFlow::CAPI::TF_DeleteTensor($$self);
  }

  sub get_type {
    my ($self) = @_;
    return ML::TensorFlow::CAPI::TF_TensorType($$self);
  }

  sub get_num_dims {
    my ($self) = @_;
    return ML::TensorFlow::CAPI::TF_NumDims($$self);
  }

  sub get_dim_size {
    my ($self, $dim_index) = @_;
    return ML::TensorFlow::CAPI::TF_Dim($$self, $dim_index);
  }

  sub get_tensor_byte_size {
    my ($self) = @_;
    return ML::TensorFlow::CAPI::TF_TensorByteSize($$self);
  }

  # WARNING: No encapsulation
  sub get_tensordata {
    my ($self) = @_;
    return ML::TensorFlow::CAPI::TF_TensorData($$self);
  }
}; # end Tensor
1;

__END__

=head1 NAME

ML::TensorFlow - ...

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

