(class_declaration
  name: (identifier) @name) @definition.class

(interface_declaration
  name: (identifier) @name) @definition.interface

(record_declaration
  name: (identifier) @name) @definition.record

(enum_declaration
  name: (identifier) @name) @definition.enum

(annotation_type_declaration
  name: (identifier) @name) @definition.annotation

(method_declaration
  name: (identifier) @name) @definition.method

(constructor_declaration
  name: (identifier) @name) @definition.constructor

(compact_constructor_declaration
  name: (identifier) @name) @definition.constructor

(annotation_type_element_declaration
  name: (identifier) @name) @definition.method

(enum_constant
  name: (identifier) @name
  body: (class_body)) @definition.constant
