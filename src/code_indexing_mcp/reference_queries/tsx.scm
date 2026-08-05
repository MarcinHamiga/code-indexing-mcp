(import_statement source: (string) @module) @reference.import
(import_statement
  (import_clause
    (named_imports
      (import_specifier name: (identifier) @name alias: (identifier) @alias)))
  source: (string) @module) @reference.import
(export_statement source: (string) @module) @reference.export
(class_heritage (_) @name) @reference.inheritance
(extends_type_clause (_) @name) @reference.inheritance
(call_expression function: (_) @name arguments: (arguments) @arguments) @reference.call
(call_expression
  function: (member_expression object: (_) @receiver property: (_) @name)
  arguments: (arguments) @arguments) @reference.call
(new_expression constructor: (_) @name arguments: (arguments) @arguments) @reference.call
(generic_type name: (_) @name) @reference.type_use
(type_annotation (_) @name) @reference.type_use
(function_declaration name: (identifier) @name parameters: (formal_parameters) @declaration.parameters)
(method_definition name: (_) @name parameters: (formal_parameters) @declaration.parameters)
