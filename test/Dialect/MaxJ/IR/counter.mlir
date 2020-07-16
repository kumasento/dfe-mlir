maxj.kernel @check_counter() -> () {
  %0 = maxj.counter 1 : i64 -> !maxj.svar<i1>
  %1 = maxj.counter 8 : i64, 32 : i64 -> !maxj.svar<i8>
}