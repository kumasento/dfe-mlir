#ifndef CONVERSION_PASS_DETAIL_H
#define CONVERSION_PASS_DETAIL_H

#include "mlir/Pass/Pass.h"

namespace dfe {
#define GEN_PASS_CLASSES
#include "dfe/Conversion/Passes.h.inc"
} // namespace dfe

#endif