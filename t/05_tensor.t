use strict;
use warnings;
use Test::More tests => 7;
use ML::TensorFlow qw(:all);
use Config qw(%Config);

SCOPE: {
  my $bytes = $Config{doublesize} * 6;
  my $blob = "\0" x $bytes;
  my $tensor = Tensor->new(
    TF_DataType_Enum()->{TF_DOUBLE},
    [2,3],
    $blob
  );

  isa_ok($tensor, "ML::TensorFlow::Tensor");
  is($tensor->get_type, TF_DataType_Enum()->{TF_DOUBLE}, "Tensor of right type");
  is($tensor->get_num_dims, 2, "Right number of dims");
  is($tensor->get_dim_size(0), 2, "Dim 0 of right size");
  is($tensor->get_dim_size(1), 3, "Dim 1 of right size");
  is($tensor->get_tensordata, $blob, "Blob unchanged"); # sigh
}

pass("Alive at end");

