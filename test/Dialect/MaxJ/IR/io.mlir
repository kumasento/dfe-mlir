"maxj.kernel" () ({
  ^body:
    %0 = maxj.const 1.0 : f64 -> !maxj.svar<i1>
    %1 = maxj.input "a", %0 : !maxj.svar<i1> -> !maxj.svar<f64>
    "maxj.terminator"() {} : () -> ()
}) {sym_name="foo", type=()->()} : () -> ()