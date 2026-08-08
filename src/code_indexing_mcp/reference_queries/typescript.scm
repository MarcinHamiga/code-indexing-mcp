(import_statement source: (string) @module) @reference.import
(import_statement
  (import_clause
    (named_imports
      (import_specifier name: (identifier) @name alias: (identifier) @alias)))
  source: (string) @module) @reference.import
(export_statement source: (string) @module) @reference.export
(export_statement) @reference.export
(class_heritage (_) @name) @reference.inheritance
(extends_type_clause (_) @name) @reference.inheritance
(decorator) @reference.decorator
(call_expression function: (_) @name arguments: (_) @arguments) @reference.call
(call_expression
  function: (member_expression object: (_) @receiver property: (_) @name)
  arguments: (_) @arguments) @reference.call
(new_expression constructor: (_) @name) @reference.call
(generic_type name: (_) @name) @reference.type_use
(type_annotation (_) @name) @reference.type_use
(member_expression) @reference.member_access
(function_declaration name: (identifier) @name parameters: (formal_parameters) @declaration.parameters)
(method_definition name: (_) @name parameters: (formal_parameters) @declaration.parameters)
(variable_declarator
  name: (identifier) @name
  value: (arrow_function parameters: (formal_parameters) @declaration.parameters))
(variable_declarator
  name: (identifier) @name
  value: (function_expression parameters: (formal_parameters) @declaration.parameters))

(identifier) @reference.identifier
(type_identifier) @reference.identifier
(shorthand_property_identifier) @reference.identifier
(shorthand_property_identifier_pattern) @reference.identifier
