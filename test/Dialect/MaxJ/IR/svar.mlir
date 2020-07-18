maxj.kernel @check_svar() -> () {
  %0 = maxj.svar : i1 
}

maxj.kernel @check_offset() -> () {
  %0 = maxj.input "a" -> !maxj.svar<i32>
  %1 = maxj.offset -1 : i64, %0 : !maxj.svar<i32>
}