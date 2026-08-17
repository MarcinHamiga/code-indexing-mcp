(preproc_def
  name: (identifier) @name) @definition.constant

(struct_specifier
  name: (type_identifier) @name) @definition.struct

(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name)) @definition.function
