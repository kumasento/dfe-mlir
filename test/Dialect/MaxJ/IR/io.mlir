// constantly take in a value and pass it to the output.
maxj.kernel @foo () -> () {
  %0 = maxj.const 1.0 : f64 -> !maxj.svar<i1>
  %1 = maxj.input "a", %0 : !maxj.svar<i1> -> !maxj.svar<f64>
  maxj.output "b", %1 : !maxj.svar<f64>, %0: !maxj.svar<i1>
}