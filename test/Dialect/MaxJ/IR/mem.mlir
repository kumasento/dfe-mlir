maxj.kernel @check_mem() -> () {
  %0 = maxj.alloc() : !maxj.mem<128xi32>
  %1 = maxj.counter 8: i64 -> !maxj.svar<i8>
  %2 = maxj.read %0 : !maxj.mem<128xi32>, %1: !maxj.svar<i8>
}