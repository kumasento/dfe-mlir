// class ContainerType<Type etype, Pred containerPred, code elementTypeCall,
//                     string descr>
// class BuildableType<code builder> {
//   // The builder call to invoke (if specified) to construct the BuildableType.
//   code builderCall = builder;
// }

// --------------------------- Fixed point types

def AnyFix : Type<CPred<"$_self.isa<FixType>()">, "MaxJ Fixed Point Type">,
             BuildableType<"FixType::get($_builder.getContext())">;

// --------------------------- SVar

class MaxJ_SVarType<list<Type> allowedTypes>
  : ContainerType<AnyTypeOf<allowedTypes>, CPred<"$_self.isa<SVarType>()">,
                  "$_self.cast<SVarType>().getUnderlyingType()", "MaxJ SVar type">;

// TODO: will support Tensor in the future.
def MaxJ_AnySVarAllowedType : AnyTypeOf<[AnyInteger, AnyFloat, AnyVector]>;

def MaxJ_AnySVarType : MaxJ_SVarType<[MaxJ_AnySVarAllowedType]>;

// ---------------------------- Memory

def MaxJ_IsMemTypePred : CPred<"$_self.isa<MemType>()">;

class MaxJ_MemOf<list<Type> allowedTypes>
  : ContainerType<AnyTypeOf<allowedTypes>, MaxJ_IsMemTypePred,
                  "$_self.cast<MemType>().getElementType()", "MaxJ Mem type">;

def MaxJ_AnyMemType : MaxJ_MemOf<[MaxJ_AnySVarAllowedType]>;