(struct_item
  name: (type_identifier) @name) @definition.struct

(enum_item
  name: (type_identifier) @name) @definition.enum

(function_item
  name: (identifier) @name) @definition.function

(const_item
  name: (identifier) @name) @definition.constant
