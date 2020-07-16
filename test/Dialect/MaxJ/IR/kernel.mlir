// a normal MaxJ kernel with a single constant output
"maxj.kernel" () ({
  ^body:
    "maxj.terminator"() {} : () -> ()
}) {sym_name="foo", type=()->()} : () -> ()