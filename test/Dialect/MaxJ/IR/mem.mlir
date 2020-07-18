maxj.kernel @check_read() -> () {
  %0 = maxj.alloc() : !maxj.mem<128xi32>
  %1 = maxj.counter 8: i64 -> !maxj.svar<i8>
  %2 = maxj.read %0 : !maxj.mem<128xi32>, %1: !maxj.svar<i8>
}

maxj.kernel @check_write() -> () {
  %0 = maxj.alloc() : !maxj.mem<128xi32>
  %1 = maxj.counter 8 : i64 -> !maxj.svar<i8>
  %2 = maxj.input "val" -> !maxj.svar<i32>
  %3 = maxj.input "enable" -> !maxj.svar<i1>
  maxj.write %0 : !maxj.mem<128xi32>, %1 : !maxj.svar<i8>, %2 : !maxj.svar<i32>, %3 : !maxj.svar<i1>
}