(type_declaration
  (type_spec
    name: (type_identifier) @name)) @definition.class

(method_declaration
  name: (field_identifier) @name) @definition.method

(function_declaration
  name: (identifier) @name) @definition.function

(const_spec
  name: (identifier) @name) @definition.constant
