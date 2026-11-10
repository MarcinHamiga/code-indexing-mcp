;; The definition is the type_spec, not the enclosing type_declaration: a grouped
;; `type ( A struct{}; B struct{} )` holds several specs under one declaration
;; node. The extractor widens a lone spec back to its declaration (`type` keyword).
(type_spec
  name: (type_identifier) @name) @definition.class

(method_declaration
  name: (field_identifier) @name) @definition.method

(function_declaration
  name: (identifier) @name) @definition.function

(const_spec
  name: (identifier) @name) @definition.constant
