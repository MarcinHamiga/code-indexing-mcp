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

;; `case north, south, east` is one enum_entry holding several names, and one node
;; can carry only one definition, so each name is its own definition. The
;; extractor widens a lone name back to its `case` entry.
(enum_entry
  name: (simple_identifier) @name @definition.constant)
