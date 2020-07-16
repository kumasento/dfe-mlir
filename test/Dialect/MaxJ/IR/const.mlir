func @lap() -> () {
  // a boolean (looks a bit weird)
  %0 = maxj.const 1. : f64 -> !maxj.svar<i1>
  %1 = maxj.const 0.283: f64 -> !maxj.svar<f64>
  %2 = maxj.const 1. : f64 { a = -65.09: f64 } -> !maxj.svar<f32>

  // return is an op from the StandardOps dialect.
  // should register them
  return 
}