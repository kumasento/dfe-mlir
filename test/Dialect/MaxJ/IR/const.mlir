func @lap() -> () {
  // integer constant
  %0 = "maxj.const"() { value = 1 : i64 } : () -> !maxj.svar<i64>

  // return is an op from the StandardOps dialect.
  // should register them
  return
}