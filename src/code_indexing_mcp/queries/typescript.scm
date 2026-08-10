(function_declaration
  name: (identifier) @name) @definition.function

(class_declaration
  name: (type_identifier) @name) @definition.class

(abstract_class_declaration
  name: (type_identifier) @name) @definition.class

(method_definition
  name: [(property_identifier) (private_property_identifier)] @name) @definition.method

(abstract_method_signature
  name: (property_identifier) @name) @definition.method

(public_field_definition
  name: (property_identifier) @name
  value: [(arrow_function) (function_expression)]) @definition.function

(variable_declarator
  name: (identifier) @name
  value: [(arrow_function) (function_expression)]) @definition.function

(interface_declaration
  name: (type_identifier) @name) @definition.interface

(type_alias_declaration
  name: (type_identifier) @name) @definition.type

(enum_declaration
  name: (identifier) @name) @definition.enum
