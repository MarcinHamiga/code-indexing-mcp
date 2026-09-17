(Decl
  (FnProto
    function: (IDENTIFIER) @name)) @definition.function

(Decl
  (VarDecl
    (IDENTIFIER) @name)) @definition.constant

(ContainerField
  (IDENTIFIER) @name) @definition.property

(TestDecl
  (STRINGLITERALSINGLE) @name) @definition.function
