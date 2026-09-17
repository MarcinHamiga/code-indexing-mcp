(class_declaration
  name: (type_identifier) @name) @definition.class

(protocol_declaration
  name: (type_identifier) @name) @definition.interface

(function_declaration
  name: (simple_identifier) @name) @definition.function

(protocol_function_declaration
  name: (simple_identifier) @name) @definition.function

(init_declaration
  "init" @name) @definition.constructor

(property_declaration
  name: (pattern
    (simple_identifier) @name)) @definition.property

(enum_entry
  (simple_identifier) @name) @definition.constant
