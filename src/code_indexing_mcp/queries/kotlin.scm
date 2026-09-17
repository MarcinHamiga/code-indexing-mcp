(class_declaration
  (type_identifier) @name) @definition.class

(object_declaration
  (type_identifier) @name) @definition.object

(function_declaration
  (simple_identifier) @name) @definition.function

(property_declaration
  (variable_declaration
    (simple_identifier) @name)) @definition.constant

(type_alias
  (type_identifier) @name) @definition.type

(enum_entry
  (simple_identifier) @name) @definition.constant
