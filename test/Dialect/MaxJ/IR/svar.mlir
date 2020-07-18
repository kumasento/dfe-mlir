maxj.kernel @check_svar() -> () {
  %0 = maxj.svar : i1 
  %1 = maxj.svar : vector<4xi8>
}

maxj.kernel @check_offset() -> () {
  %0 = maxj.input "a" -> !maxj.svar<i32>
  %1 = maxj.offset -1 : i64, %0 : !maxj.svar<i32>

  %v0 = maxj.input "vec_a" -> !maxj.svar<vector<8xi32>>
  %v1 = maxj.offset -1 : i64, %v0 : !maxj.svar<vector<8xi32>>
}

maxj.kernel @check_conn() -> () {
  %0 = maxj.svar : i32
  %1 = maxj.input "a" -> !maxj.svar<i32>
  maxj.conn %1, %0 : !maxj.svar<i32>
}