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