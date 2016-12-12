use strict;
use warnings;
use Test::More tests => 11;
use ML::TensorFlow qw(:all);

SCOPE: {
  my $buffer = Buffer->new;
  isa_ok($buffer, "ML::TensorFlow::Buffer");
}
pass("Alive after empty buffer DESTROY");

SCOPE: {
  my $buffer = Buffer->new("foo");
  isa_ok($buffer, "ML::TensorFlow::Buffer");
  is($buffer->get_data_copy, "foo", "Buffer->get_data_copy");
  is($buffer->get_data_view, "foo", "Buffer->get_data_view");
}
pass("Alive after non-empty buffer DESTROY");

SCOPE: {
  my $buffer = Buffer->new("foo");

  is($buffer->get_data_copy, "foo", "Buffer->get_data_copy");
  is($buffer->get_data_view, "foo", "Buffer->get_data_view");

  my $new_data = "barbazbuz";
  $buffer->set_data($new_data);
  is($buffer->get_data_copy, "$new_data", "Buffer->get_data_copy after setting data");
  is($buffer->get_data_view, "$new_data", "Buffer->get_data_view after setting data");
}

pass("Alive at end");

