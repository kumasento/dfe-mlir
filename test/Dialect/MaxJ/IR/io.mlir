"maxj.kernel" () ({
  ^body:
    %0 = maxj.const 1.0 : f64 -> !maxj.svar<i1>
    "maxj.terminator"() {} : () -> ()
}) {sym_name="foo", type=()->()} : () -> ()