use strict;
use warnings;
use Test::More tests => 5;
use ML::TensorFlow qw(:all);

my $s = Status->new;
isa_ok($s, "ML::TensorFlow::Status");

is($s->get_code, TF_Code_Enum()->{TF_OK}, "Initialized to TF_OK");
is($s->get_message, "", "Initialized to empty string");

$s->set_status(TF_Code_Enum()->{TF_CANCELLED}, "cancel");

is($s->get_code, TF_Code_Enum()->{TF_CANCELLED}, "Setting code works");
is($s->get_message, "cancel", "Setting msg works");

