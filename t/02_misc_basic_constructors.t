use strict;
use warnings;
use Test::More tests => 3;
use ML::TensorFlow qw(:all);

my $sessopt = SessionOptions->new;
isa_ok($sessopt, "ML::TensorFlow::SessionOptions");

my $graph = Graph->new;
isa_ok($graph, "ML::TensorFlow::Graph");

my $session = Session->new($graph, $sessopt);
isa_ok($session, "ML::TensorFlow::Session");



